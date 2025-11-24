#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

#include <iostream>
#include <cmath>
#include <cstring>

using namespace std;

//Grayscale 
#define RED_WEIGHT   0.2989
#define GREEN_WEIGHT 0.5870
#define BLUE_WEIGHT  0.1140

//Canny
#define WHITE 255
#define BLACK 0
#define NON_RELEVANT 0
#define WEAK 1
#define STRONG 2
#define CANNY 0.25
#define M_PI 3.14159265358979323846

//FloyedSteinberg
#define COMPRESSED 16
#define ALPHA 7/16.0
#define BETA 3/16.0
#define GAMMA 5/16.0
#define DELTA 1/16.0



//1.Grayscale
unsigned char* toGrayscale(unsigned char* img, int width, int height){
    unsigned char* gray = new unsigned char[width * height];
    for (int i = 0; i < width * height; i++){
        int idx = i * 4;
        gray[i] = img[idx] * RED_WEIGHT + img[idx + 1] * GREEN_WEIGHT + img[idx + 2] * BLUE_WEIGHT;
    }
    return gray;
}


//2.Canny
void applyKernel(float *buffer, float *newBuffer, int width, int x, int y,float *kernel, int kernelwidth, int kernelheight, float norm)
{
    float sum = 0;
    int hw = (kernelwidth - 1) / 2;
    int hh = (kernelheight - 1) / 2;

    for (int i = -hh; i <= hh; i++)
        for (int j = -hw; j <= hw; j++)
            sum += buffer[(y+i)*width + (x+j)] *
                   kernel[(i+hh)*kernelwidth + (j+hw)];

    newBuffer[y*width + x] = sum * norm;
}


float *convolution(float *buffer, float *newBuffer, int width, int height,float *kernel, int kwidth, int kheight, float norm)
{
    memcpy(newBuffer, buffer, width*height*sizeof(float));

    for (int y = 1; y < height-1; y++)
        for (int x = 1; x < width-1; x++)
            applyKernel(buffer, newBuffer, width, x, y, kernel, kwidth, kheight, norm);

    return newBuffer;
}


float clip(float pixel)
{
    if (pixel > WHITE) {
        return WHITE;
    }
    if (pixel < BLACK) {
        return BLACK;
    }
    return pixel;
}


//2.1.Gradient Calculation
void computeGradient(float* fBuffer, int width, int height,float* blurred, float* xConv, float* yConv,unsigned char* gradient, float* angles, float scale)
{
    float xSobel[] = {1,0,-1, 2,0,-2, 1,0,-1};
    float ySobel[] = {1,2,1, 0,0,0, -1,-2,-1};
    float gaussian[] = {1,2,1, 2,4,2, 1,2,1};

    convolution(fBuffer, blurred, width, height, gaussian, 3, 3, 1.0/16);

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
            {
                gradient[x + y * width] = 0;
                angles[x + y * width] = 0;
                continue;
            }

            applyKernel(blurred, xConv, width, x, y, xSobel, 3, 3, scale);
            applyKernel(blurred, yConv, width, x, y, ySobel, 3, 3, scale);

            float gx = xConv[y*width + x];
            float gy = yConv[y*width + x];

            gradient[y*width + x] = clip(sqrt(gx*gx + gy*gy));
            angles[y*width + x] = atan2(gy, gx) * 180 / M_PI;
        }
}


// 2.2.Non-Max Suppression
void nonMaxSuppression(unsigned char* gradient, float* angles,unsigned char* outlined, int width, int height)
{
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            float angle = angles[y*width + x];
            if (angle < 0) angle += 360;

            int dx = 0, dy = 0;

            if (angle <= 22.5 || angle > 337.5 || (angle > 157.5 && angle <= 202.5))
                dx = 1, dy = 0;

            else if ((angle > 22.5 && angle <= 67.5) || (angle > 202.5 && angle <= 247.5))
                dx = 1, dy = 1;   

            else if ((angle > 67.5 && angle <= 112.5) || (angle > 247.5 && angle <= 292.5))
                dx = 0, dy = 1;

            else if ((angle > 112.5 && angle <= 157.5) || (angle > 292.5 && angle <= 337.5))
                dx = -1, dy = 1;

            unsigned char curr = gradient[y*width + x];
            unsigned char p1 = gradient[(y+dy)*width + (x+dx)];
            unsigned char p2 = gradient[(y-dy)*width + (x-dx)];

            if (p1<curr && p2<curr ){
                outlined[y*width + x]=curr;
            }
            else{
                outlined[y*width + x]=BLACK;
            }
        }
}


//2.3.Double Thresholding
void doubleThreshold(unsigned char* outlined, int* strength,int width, int height, float lower, float upper)
{
    int lowT = lower * WHITE;
    int highT = upper * WHITE;

    for (int y = 1; y < height-1; y++){
        for (int x = 1; x < width-1; x++){
            unsigned char p = outlined[y*width + x];
            if (p < lowT) {
                strength[y*width + x] = NON_RELEVANT;
            }
            else if (p <= highT) {
                strength[y*width + x] = WEAK;
            }
            else{
                strength[y*width + x] = STRONG;
            }
        }
    }
}


//2.4.Hysteresis
void hysteresis(int* strength, unsigned char* outlined,int width, int height)
{
    for (int y = 1; y < height-1; y++)
        for (int x = 1; x < width-1; x++)
        {
            if (strength[y*width + x] == NON_RELEVANT)
                outlined[y*width + x] = BLACK;
            if (strength[y*width + x] == STRONG)
                outlined[y*width + x] = WHITE;

            if (strength[y*width + x] == WEAK)
            {
                if(
                    strength[(y-1)*width + (x-1)] == STRONG ||
                    strength[(y-1)*width + x]     == STRONG ||
                    strength[(y-1)*width + (x+1)] == STRONG ||
                    strength[y*width + (x-1)]     == STRONG ||
                    strength[y*width + (x+1)]     == STRONG ||
                    strength[(y+1)*width + (x-1)] == STRONG ||
                    strength[(y+1)*width + x]     == STRONG ||
                    strength[(y+1)*width + (x+1)] == STRONG)
                    {
                        outlined[y*width + x] =  WHITE ;
                        strength[y*width + x] =  BLACK ;

                    }
            else
                outlined[y*width + x] = BLACK;
            }
        }
}

//main function to canny 
unsigned char* applyCanny(unsigned char* gray, int w, int h)
{
    // Allocate buffers
    float* fBuffer = new float[w * h];
    float* blurred = new float[w * h];
    float* xConv = new float[w * h];
    float* yConv = new float[w * h];

    unsigned char* gradient  = new unsigned char[w * h];
    float* angles = new float[w * h];
    unsigned char* nmsOutput = new unsigned char[w * h];
    int* strength = new int[w * h];
    unsigned char* thresholdImg = new unsigned char[w * h];
    unsigned char* finalOutput  = new unsigned char[w * h];

    // Copy gray 
    for (int i = 0; i < w * h; i++)
        fBuffer[i] = gray[i];

    // a. Gradient
    computeGradient(fBuffer, w, h, blurred, xConv, yConv, gradient, angles, CANNY);
    stbi_write_png("res/textures/2.1.Canny_Gradient.png", w, h, 1, gradient, w);

    // b. Non-Max Suppression
    nonMaxSuppression(gradient, angles, nmsOutput, w, h);
    stbi_write_png("res/textures/2.2.Canny_NMS.png", w, h, 1, nmsOutput, w);

    // c. Double Threshold
    doubleThreshold(nmsOutput, strength, w, h, 0.1f, 0.15f);

    for (int i = 0; i < w * h; i++) {
        if (strength[i] == STRONG)       thresholdImg[i] = 255;
        else if (strength[i] == WEAK)    thresholdImg[i] = 128;
        else                              thresholdImg[i] = 0;
    }
    stbi_write_png("res/textures/2.3.Canny_Threshold.png", w, h, 1, thresholdImg, w);

    // d. Hysteresis
    hysteresis(strength, finalOutput, w, h);
    stbi_write_png("res/textures/2.4.Canny_Hysteresis.png", w, h, 1, finalOutput, w);

    delete[] fBuffer;
    delete[] blurred;
    delete[] xConv;
    delete[] yConv;
    delete[] gradient;
    delete[] angles;
    delete[] nmsOutput;
    delete[] strength;
    delete[] thresholdImg;

    return finalOutput;
}


// 3. Halftone
unsigned char *halftone(unsigned char *buffer, int width, int height)
{
    int new_width = width * 2;
    int new_height = height * 2;

    // allocate with malloc --> width*2 * height*2
    unsigned char *result = (unsigned char *) malloc(new_width * new_height * sizeof(unsigned char));
    if (!result) return NULL; // malloc failed

    for (int i = 0; i < width * height; i++)
    {
        int row = i / width;
        int col = i % width;

        // starting pixel in the 2x2 block
        int base = (row * 2) * new_width + (col * 2);

        unsigned char p = buffer[i];

        if (p < 51)
        {
            result[base] = result[base+1] =
            result[base+new_width] = result[base+new_width+1] = BLACK;
        }
        else if (p < 102)
        {
            result[base] = BLACK;
            result[base+1] = BLACK;
            result[base+new_width] = WHITE;
            result[base+new_width+1] = BLACK;
        }
        else if (p < 153)
        {
            result[base] = WHITE;
            result[base+1] = BLACK;
            result[base+new_width] = WHITE;
            result[base+new_width+1] = BLACK;
        }
        else if (p < 204)
        {
            result[base] = BLACK;
            result[base+1] = WHITE;
            result[base+new_width] = WHITE;
            result[base+new_width+1] = WHITE;
        }
        else
        {
            result[base] = result[base+1] =
            result[base+new_width] = result[base+new_width+1] = WHITE;
        }
    }

    return result;
}



//4.floyedSteinberg
unsigned char *floyedSteinberg(unsigned char *buffer, int width, int height,float a, float b, float c, float d)
{
    unsigned char *diff = new unsigned char[width * height];
    unsigned char *result = new unsigned char[width * height];

    memcpy(diff, buffer, width*height);

    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
        {
            int idx = y*width + x;

            unsigned char oldP = diff[idx];
            unsigned char newP = (unsigned char)((oldP / 255.0 * COMPRESSED) / COMPRESSED * 255.0);
            result[idx] = newP;

            float error = oldP - newP;

            if (x+1 < width)
                diff[idx+1] += error * a;

            if (y+1 < height)
            {
                if (x > 0) diff[(y+1)*width + (x-1)] += error * b;
                diff[(y+1)*width + x] += error * c;
                if (x+1 < width) diff[(y+1)*width + (x+1)] += error * d;
            }
        }

    delete[] diff;
    return result;
}


//text files for all the 4 images
void saveGrayscale(const char* filename, unsigned char* gray, int w, int h)
{
    FILE* f = fopen(filename, "w");
    if (!f) return;

    int size = w * h;
    for (int i = 0; i < size; i++)
    {
        float v = gray[i] / 255.0f;   
        fprintf(f, "%f", v);

        if (i < size - 1) fprintf(f, ",");
    }
    fclose(f);
}

void saveCanny(const char* filename, unsigned char* out, int w, int h)
{
    FILE* f = fopen(filename, "w");
    if (!f) return;

    int size = w * h;
    for (int i = 0; i < size; i++)
    {
        float v = (out[i] == 255 ? 1.0f : 0.0f); 
        fprintf(f, "%d", (int)v);

        if (i < size - 1) fprintf(f, ",");
    }
    fclose(f);
}

void saveHalftone(const char* filename, unsigned char* half, int w, int h)
{
    FILE* f = fopen(filename, "w");
    if (!f) return;

    int size = w * h;
    for (int i = 0; i < size; i++)
    {
        int v = (half[i] > 128 ? 1 : 0);
        fprintf(f, "%d", v);

        if (i < size - 1) fprintf(f, ",");
    }
    fclose(f);
}

void saveFloyd(const char* filename, unsigned char* fs, int w, int h)
{
    FILE* f = fopen(filename, "w");
    if (!f) return;

    int size = w * h;
    for (int i = 0; i < size; i++)
    {
        int v = (fs[i] / 255.0f) * 15;  
        fprintf(f, "%d", v);

        if (i < size - 1) fprintf(f, ",");
    }
    fclose(f);
}



// ===================== MAIN =======================
int main()
{
    int w, h, c;
    unsigned char* img = stbi_load("res/textures/Lenna.png", &w, &h, &c, 4);

    // 1. Grayscale
    unsigned char* gray = toGrayscale(img, w, h);
    stbi_write_png("res/textures/1.Gray.png", w, h, 1, gray, w);
    // saveGrayscale("Grayscale.txt", gray, w, h);

    // 2.CANNY 
    unsigned char* canny = applyCanny(gray, w, h);
    stbi_write_png("res/textures/2.Canny.png", w, h, 1, canny, w);
    // saveCanny("Canny.txt", canny, w, h);

    // 3. Halftone
    unsigned char *halftoneBuff = halftone(gray, w, h);
    stbi_write_png("res/textures/3.Halftone.png", w*2, h*2, 1, halftoneBuff, w*2);
    // saveHalftone("Halftone.txt", halftoneBuff, w*2, h*2);

    // 4. FloydSteinberg
    unsigned char *fs = floyedSteinberg(gray, w, h, ALPHA, BETA, GAMMA, DELTA);
    stbi_write_png("res/textures/4.FloydSteinberg.png", w, h, 1, fs, w);
    // saveFloyd("FloyedSteinberg.txt", fs, w, h);

    cout << "Done!" << endl;
    return 0;
}
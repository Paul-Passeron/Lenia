#pragma once
#include "raylib.h"
#include "rlgl.h"
#include "raymath.h"

#include <math.h>
#include <fftw3.h>

#define REAL 0
#define IMAG 1
const int GRID_SIZE = 100;
// grid size must be odd for kernel reasons
const int B = 3;

using namespace std;
float norm_array2d(int n0, int n1, fftw_complex a[][GRID_SIZE])
{
    float acc = 0;
    for (int i = 0; i < n0; i++)
    {
        for (int j = 0; j < n1; j++)
        {
            float r = pow(a[i][j][REAL], 2) + pow(a[i][j][IMAG], 2);
            acc += sqrt(r);
        }
    }
    return acc;
}

float random_state()
{
    return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

float float_mod(float a, float b)
{
    return fmod(a, b);
}

float euc_dist(float x, float y)
{
    return sqrt(x * x + y * y);
    // return max(abs(x), abs(y));
}

bool is_in_neighborhood(float r, float x, float y)
{
    return euc_dist(x, y) <= r;
}

int emod(int a, int b)
{
    if (a < 0)
    {
        return b + a;
    }
    else if (a > b)
    {
        return a - b;
    }
    return a;
}

float rect(float x, float a, float b)
{
    if (a <= x && x <= b)
    {
        return 1;
    }
    return 0;
}

float clip(float a)
{
    if (a > 1)
    {
        return 1;
    }
    if (a < 0)
    {
        return 0;
    }
    return a;
}

float max(float a, float b)
{
    if (a > b)
    {
        return a;
    }
    return b;
}

float min(float a, float b)
{
    if (a > b)
    {
        return b;
    }
    return a;
}

void complex_mul(fftw_complex a, fftw_complex b, fftw_complex c, float fact)
{
    c[REAL] = fact * (a[REAL] * b[REAL] - a[IMAG] * b[IMAG]);
    c[IMAG] = fact * (a[REAL] * b[IMAG] + a[IMAG] * b[REAL]);
}

void fft_product_lenia(fftw_complex a[][GRID_SIZE], fftw_complex b[][GRID_SIZE], fftw_complex c[][GRID_SIZE])
{
    float factor =  1.0f/norm_array2d(GRID_SIZE, GRID_SIZE, a);
    for (int i = 0; i < GRID_SIZE; i++)
    {
        for (int j = 0; j < GRID_SIZE; j++)
        {
            complex_mul(a[i][j], b[i][j], c[i][j], factor);
        }
    }
}

float g_exp(float u, float mu, float sigma)
{
    float a = -powf(u - mu, 2) / (2.0f * powf(sigma, 2));
    return 2.0f * expf(a) - 1.0f;
}

float g_rect(float r, float mu, float sigma)
{
    return 2.0 * rect(r, mu - sigma, mu + sigma) - 1;
}

float k_c_exp(float r, float alpha)
{
    return expf(alpha - alpha / (4.0f * r * (1.0f - r)));
}

float k_c_pol(float r, float alpha)
{
    return pow(4.0 * r * (1.0 - r), alpha);
}

float k_c_gol(float r)
{
    return rect(r, 0.25, 0.75) + 0.5 * rect(r, 0, 0.25);
}

Vector4 createVector4(float x, float y, float z, float w)
{
    Vector4 res;
    res.x = x;
    res.y = y;
    res.z = z;
    res.w = w;
    return res;
}

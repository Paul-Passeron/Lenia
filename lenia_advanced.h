#pragma once

#include "lenia_core.h"
#include "lenia_math.h"

#include <vector>

using namespace std;

struct Grid
{
  fftw_complex t[GRID_SIZE][GRID_SIZE];
};

struct Param
{
  int B;
  vector<float> beta;
  float r;
  float alpha;
  float mu;
  float sigma;
  float dt;
};

class AdvancedLenia
{

private:
  int channels;
  vector<vector<float>> channel_mixer;

  vector<Grid> grid;
  Grid kernel;
  Param params;
  float k_s_norm_inv;

  void init_blank_grid()
  {
    for (int i = 0; i < channels; i++)
    {
      Grid temp;
      grid.push_back(temp);
    }
  }

  float g(float r) { return g_exp(r, mu, sigma); }

  float k_c(float r) { return k_c_exp(r, alpha); }

  float k_s(int i, int j)
  {
    float a = euc_dist(i, j) / params.r;
    int index = emod(floor((float)params.B * a), params.B);
    float u = float_mod((float)params.B * a, 1.0);
    return beta[index] * k_c(u);
  }

  void init_k_s_norm()
  {
    float acc = 0;
    for (int i = 0; i < GRID_SIZE; i++)
    {
      for (int j = 0; j < GRID_SIZE; j++)
      {
        if (is_in_neighborhood(r, i, j))
        {
          acc += k_s(i, j);
        }
      }
    }
    k_s_norm_inv = 1.0f / acc;
  }

  float k(int i, int j) { return k_s(i, j) * k_s_norm_inv; }

  void init_kernel()
  {
    for (int i = 0; i < GRID_SIZE; i++)
    {
      for (int j = 0; j < GRID_SIZE; j++)
      {
        int x = i - GRID_SIZE / 2;
        int y = j - GRID_SIZE / 2;
        kernel[i][j] = k(x, y);
      }
    }
  }

  void convol(fftw_complex gri[][GRID_SIZE], fftw_complex conv[][GRID_SIZE])
  {
    fftw_complex grid_fft[GRID_SIZE][GRID_SIZE];
    fftw_complex kernel_fft[GRID_SIZE][GRID_SIZE];
    fftw_complex product_fft[GRID_SIZE][GRID_SIZE];
    fftw_plan plan_grid = fftw_plan_dft_2d(GRID_SIZE, GRID_SIZE, &gri[0][0], &grid_fft[0][0], FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan plan_kernel = fftw_plan_dft_2d(GRID_SIZE, GRID_SIZE, &kernel[0][0], &kernel_fft[0][0], FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan_grid);
    fftw_execute(plan_kernel);
    fftw_destroy_plan(plan_grid);
    fftw_destroy_plan(plan_kernel);
    fft_product_lenia(grid_fft, kernel_fft, product_fft);
    fftw_plan plan_conv = fftw_plan_dft_2d(GRID_SIZE, GRID_SIZE, &product_fft[0][0], &conv[0][0], FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(plan_conv);
    fftw_destroy_plan(plan_conv);
    fftw_cleanup();
  }

  void convol_channel(int chan, fftw_complex conv[][GRID_SIZE])
  {
    convol(grid.at(chan), conv);
  }

  void init_channel_vector(vector<Grid> vec)
  {
    for (int i = 0; i < channels; i++)
    {
      Grid temp;
      vec.push_back(temp);
    }
  }

  void add_grids(fftw_complex &a[][GRID_SIZE], fftw_complex b[][GRID_SIZE])
  {
    for (int i = 0; i < GRID_SIZE; i++)
    {
      for (int j = 0; j < GRID_SIZE; j++)
      {
        a[i][j] += b[i][j];
      }
    }
  }

  vector<Grid> compute_convolutions()
  {
    vector<Grid> res;
    init_channel_vector(res);
    for (int in_channel = 0; in_channel < channels; in_channel++)
    {
      for (int out_channel = 0; out_channel < channels; out_channel++)
      {
        float coeff = channel_mixer.at(in_channel).at(out_channel);
        Grid temp;
        convol_channel(in_channel, temp);
        add_grids(res.at(out_channel), temp); 
      }
    }
  }

public:
  AdvancedLenia(Param param, int n, vector<vector<float>> mixer)
  {
    channels = n;
    params = param;
    channel_mixer = mixer;
  }
};
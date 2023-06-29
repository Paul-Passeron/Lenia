#pragma once

#include "lenia_math.h"

class Lenia
{
private:
  fftw_complex grid[GRID_SIZE][GRID_SIZE];
  fftw_complex kernel[GRID_SIZE][GRID_SIZE];
  fftw_complex to_test[GRID_SIZE][GRID_SIZE];
  float beta[B];
  float r;
  float alpha;
  float mu;
  float sigma;
  float k_s_norm;
  float dt;

  float g(float r)
  {
    
    return g_exp(r, mu, sigma);
  }

  float k_c(float r)
  {
    return k_c_exp(r, alpha);
  }

  float k_s(int i, int j)
  {
    float a = euc_dist(i, j) / r;
    int index = emod(floor((float)B * a), B);
    float u = float_mod((float)B * a, 1.0);
    return beta[index] * k_c(u);
  }

  float init_k_s_norm()
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
    return acc;
  }

  float k(int i, int j)
  {
    return k_s(i, j) / k_s_norm;
  }

  void set_grid_blank()
  {
    for (int i = 0; i < GRID_SIZE; i++)
    {
      for (int j = 0; j < GRID_SIZE; j++)
      {
        grid[i][j][REAL] = 0;
        grid[i][j][IMAG] = 0;
      }
    }
  }

  void init_kernel()
  {
    for (int i = -GRID_SIZE / 2; i < GRID_SIZE / 2; i++)
    {
      for (int j = -GRID_SIZE / 2; j < GRID_SIZE / 2; j++)
      {
        int x = emod(i, GRID_SIZE);
        int y = emod(j, GRID_SIZE);
        if (is_in_neighborhood(r, i, j))
        {
          kernel[x][y][REAL] = k(i, j);
          kernel[x][y][IMAG] = 0;
        }
      }
    }
  }

  void convol(fftw_complex conv[][GRID_SIZE])
  {
    fftw_complex grid_fft[GRID_SIZE][GRID_SIZE];
    fftw_complex kernel_fft[GRID_SIZE][GRID_SIZE];
    fftw_complex product_fft[GRID_SIZE][GRID_SIZE];
    fftw_plan plan_grid = fftw_plan_dft_2d(GRID_SIZE, GRID_SIZE, &grid[0][0], &grid_fft[0][0], FFTW_FORWARD, FFTW_ESTIMATE);
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

public:
  Lenia(float r_, float alpha_, float mu_, float sigma_, float *beta_, float dt_)
  {
    r = r_;
    alpha = alpha_;
    mu = mu_;
    sigma = sigma_;
    dt = dt_;
    for (int i = 0; i < B; i++)
    {
      beta[i] = beta_[i];
    }
    set_grid_blank();
    k_s_norm = init_k_s_norm();
    init_kernel();
  }

  void set_grid_random()
  {
    for (int i = GRID_SIZE/6; i < 4*GRID_SIZE/6; i++)
    {
      for (int j = GRID_SIZE/6; j < GRID_SIZE/2.5; j++)
      {
        grid[i][j][REAL] = pow(random_state(), 0.3);
        grid[i][j][IMAG] = 0;
      }
    }
    for (int i = GRID_SIZE/2; i < 5*GRID_SIZE/6; i++)
    {
      for (int j = GRID_SIZE/2; j < 5*GRID_SIZE/6; j++)
      {
        grid[i][j][REAL] = pow(random_state(), 0.3);
        grid[i][j][IMAG] = 0;
      }
    }
  }

  float get_cell_state(int i, int j)
  {
    int x = emod(i, GRID_SIZE);
    int y = emod(j, GRID_SIZE);
    return grid[x][y][REAL];
  }

  float get_kernel_state(int i, int j)
  {
    int x = emod(i, GRID_SIZE);
    int y = emod(j, GRID_SIZE);
    return kernel[x][y][REAL];
  }

  float get_test_state(int i, int j)
  {
    int x = emod(i, GRID_SIZE);
    int y = emod(j, GRID_SIZE);
    return to_test[x][y][REAL];
  }

  void print_kernel()
  {
    for (int i = 0; i < GRID_SIZE; i++)
    {
      for (int j = 0; j < GRID_SIZE; j++)
      {
        std::cout << kernel[i][j][REAL] << "  ";
      }
      std::cout << endl;
    }
  }
  void test_fftw()
  {
    fftw_plan plan_kernel = fftw_plan_dft_2d(GRID_SIZE, GRID_SIZE, &kernel[0][0], &to_test[0][0], FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan_kernel);
    fftw_destroy_plan(plan_kernel);
    fftw_cleanup();
  }

  void popul_grid(float a[][29], int n0, int n1)
  {
    set_grid_blank();
    int startx = (GRID_SIZE - n0) / 2;
    int starty = (GRID_SIZE - n1) / 2;
    for (int i = 0; i < n0; i++)
    {
      for (int j = 0; j < n1; j++)
      {
        grid[startx + i][starty + j][REAL] = a[i][j];
      }
    }
  }

  void update()
  {
    fftw_complex to_process[GRID_SIZE][GRID_SIZE];
    convol(to_process);
    for (int i = 0; i < GRID_SIZE; i++)
    {
      for (int j = 0; j < GRID_SIZE; j++)
      {
        grid[i][j][REAL] = clip(grid[i][j][REAL] + dt * g(to_process[i][j][REAL]));
        // grid[i][j][REAL] = to_process[i][j][REAL];
        grid[i][j][IMAG] = clip(grid[i][j][IMAG] + dt * g(to_process[i][j][IMAG]));
      }
    }
  }
};

void displayGrid(int w, int h, Lenia lenia)
{
  float dx = (float)w / (float)GRID_SIZE;
  float dy = (float)h / (float)GRID_SIZE;
  for (int i = 0; i < GRID_SIZE; i++)
  {
    for (int j = 0; j < GRID_SIZE; j++)
    {
      float v = lenia.get_cell_state(i, j);
      Vector4 col_norm;
      col_norm.x = v/2;
col_norm.y = v/5;
      col_norm.z = v/5;
      col_norm.w = 1.0;
      Color col = ColorFromNormalized(col_norm);
      DrawRectangle((int)((float)i * dx), (int)((float)j * dy), ceil(dx), ceil(dy), col);
    }
  }
}

void displayKernel(int w, int h, Lenia lenia)
{
  float dx = (float)w / (float)GRID_SIZE;
  float dy = (float)h / (float)GRID_SIZE;
  for (int i = 0; i < GRID_SIZE; i++)
  {
    for (int j = 0; j < GRID_SIZE; j++)
    {
      float v = lenia.get_kernel_state(i, j);
      Vector4 col_norm;
      col_norm.x = v;
      col_norm.y = v/10;
      col_norm.z = v/10;
      col_norm.w = 1.0;
      Color col = ColorFromNormalized(col_norm);
      DrawRectangle((int)((float)i * dx), (int)((float)j * dy), ceil(dx), ceil(dy), col);
    }
  }
}

void displayTest(int w, int h, Lenia lenia)
{
  float dx = (float)w / (float)GRID_SIZE;
  float dy = (float)h / (float)GRID_SIZE;
  for (int i = 0; i < GRID_SIZE; i++)
  {
    for (int j = 0; j < GRID_SIZE; j++)
    {
      float v = lenia.get_test_state(i, j);
      Vector4 col_norm;
      col_norm.x = v;
      col_norm.y = v;
      col_norm.z = v;
      col_norm.w = 1.0;
      Color col = ColorFromNormalized(col_norm);
      DrawRectangle((int)((float)i * dx), (int)((float)j * dy), ceil(dx), ceil(dy), col);
    }
  }
}

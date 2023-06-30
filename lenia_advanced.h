#pragma once

#include "lenia_core.h"
#include "lenia_math.h"

#include <vector>

using namespace std;


struct Grid {
  fftw_complex t[GRID_SIZE][GRID_SIZE];
};

struct Param {
  int B;
  vector<float> beta;
  float r;
  float alpha;
  float mu;
  float sigma;
  float dt;
};

class AdvancedLenia {
  
  private:
  int channels;
  vector<Grid> grid;
  Param params;

  public:
  AdvancedLenia(Param param, int n){
    channels = n;
    params = param;
  }
};
gio trash fftw_ex
clear
g++ -o fftw_ex main.cpp -lfftw3 -lraylib -lGL -lm -lpthread -ldl -lrt -lX11
./fftw_ex

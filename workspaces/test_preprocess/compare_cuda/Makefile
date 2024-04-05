CUDA_INC = /usr/local/cuda/include

pro : bilinear_interpolation.o load_img.o
	nvcc $^ $(LDFLAGS) -o $@

load_img.o : load_img.cpp
	g++ -Iinclude -I $(CUDA_INC) -c load_img.cpp -o load_img.o

bilinear_interpolation.o : bilinear_interpolation.cu
	nvcc -Iinclude -I $(CUDA_INC) -c bilinear_interpolation.cu -o bilinear_interpolation.o

run : pro
	  ./pro
	  python compare.py

clean : 
	rm -f load_img.o bilinear_interpolation.o pro

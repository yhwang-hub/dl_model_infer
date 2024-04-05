pngs := $(shell find . -name "*.png")
jpgs := $(shell find . -name "*.jpg")

pro1 : load_img.cpp
	g++ -Iinclude $< -o $@

run : pro1
	./pro1
	python compare.py

clean : 
	rm -f pro1 $(pngs) $(jpgs)
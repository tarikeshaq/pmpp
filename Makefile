.PHONY: chapter4 clean


chapter3: chapter3/gray.cu
	nvcc chapter3/gray.cu -o gray
	./gray


chapter4: chapter4/prop.cu
	nvcc chapter4/prop.cu -o prop
	./prop


clean: 
	rm -f prop
	rm -f gray

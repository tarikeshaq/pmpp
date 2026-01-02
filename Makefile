.PHONY: chapter4 chapter3 chapter5 clean


chapter3: chapter3/gray.cu
	nvcc chapter3/gray.cu -o gray
	./gray


chapter4: chapter4/prop.cu
	nvcc chapter4/prop.cu -o prop
	./prop

chapter5: chapter5/mem.cu
	nvcc -lcublas chapter5/mem.cu -o mem
	./mem


clean: 
	rm -f prop
	rm -f gray
	rm -f mem

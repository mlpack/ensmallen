rosenbrock: rosenbrock.cpp
	g++ -std=c++11 -O3 -o rosenbrock rosenbrock.cpp -I../../../include/ -larmadillo

linear_regression: linear_regression.cpp
	g++ -std=c++11 -O3 -o linear_regression linear_regression.cpp -I../../../include/ -larmadillo

linear_regression_two_functions: linear_regression_two_functions.cpp
	g++ -std=c++11 -O3 -o linear_regression_two_functions linear_regression_two_functions.cpp -I../../../include/ -larmadillo

run: rosenbrock linear_regression linear_regression_two_functions
	./run_rosenbrock.sh
	./run_linear_regression.sh

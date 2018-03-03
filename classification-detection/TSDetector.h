#include <Python.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/types_c.h"
#include "opencv2/opencv_modules.hpp"
#include <iostream>
#include <numpy/ndarrayobject.h>
#include "pyboost_cv3_converter.cpp"
#include <chrono>
#include "stdlib.h"
using namespace pbcvt;
using namespace cv;
using namespace std;

class TSDetector {
	PyObject *pName, *pModule, *pDict, *init_predict, *close_predict, *predict;
	PyObject *pArgs, *pValue;

	public:
		TSDetector(string python_path);
		~TSDetector();
		int detect(const Mat& m);
};
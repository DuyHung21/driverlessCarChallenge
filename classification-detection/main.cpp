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


	using namespace pbcvt;
	using namespace cv;
	using namespace std;

	PyObject *pName, *pModule, *pDict, *init_predict, *close_predict, *predict;
	PyObject *pArgs, *pValue;
	char* scriptName = "predictTS";

	void init() {
		char* scriptName = "predictTS";

	    //Init
	    Py_Initialize();

	    //Get python script name
	    pName = PyUnicode_DecodeFSDefault(scriptName);


	    //Import python
	    pModule = PyImport_Import(pName);
	    Py_DECREF(pName);


	    if (pModule != NULL) {
	    	init_predict = PyObject_GetAttrString(pModule, "init");
	    	PyObject_CallObject(init_predict, NULL);
	    	Py_DECREF(init_predict);
	    }
	}

	int detect(Mat m) {
		int result;
	    if (pModule != NULL) {
	    	
	    	predict = PyObject_GetAttrString(pModule, "detect");
	    	close_predict = PyObject_GetAttrString(pModule, "finalize");

	    	if (!(predict && PyCallable_Check(predict))) {
	    		if (PyErr_Occurred())
	                PyErr_Print();
	            fprintf(stderr, "Cannot find function \"%s\"\n", "detect");
	            result = -1;
	    	}	
	    	else {
	    		Py_DECREF(init_predict);
	    		pArgs = PyTuple_New(1);
	    		pValue = fromMatToNDArray(m);
	    		if (!pValue) {
	    			Py_DECREF(predict);
	    			Py_DECREF(close_predict);
	    			return -1;
	    		}
	    		PyTuple_SetItem(pArgs, 0, pValue);
	    		pValue = PyObject_CallObject(predict, pArgs);
	    		if (pValue != NULL) {
	                result = PyLong_AsLong(pValue);
	                Py_DECREF(pValue);
	            } else {
	            	Py_DECREF(predict);
	            	Py_DECREF(close_predict);
	                Py_DECREF(pModule);
	                PyErr_Print();
	                fprintf(stderr,"Call failed\n");
	                result = -1;
	            }
	    	}
	    } else {
	        PyErr_Print();
	        fprintf(stderr, "Failed to load \"%s\"\n", "predictTS");
	        result = -1;
	    }
	    return result;
	}	

	int finish() {
		if (!(close_predict && PyCallable_Check(close_predict))) {
	    		if (PyErr_Occurred())
	                PyErr_Print();
	            fprintf(stderr, "Cannot find function \"%s\"\n", "finalize");
	    }
		else {
			PyObject_CallObject(close_predict, NULL);
		}			                


	    Py_XDECREF(init_predict);
	    Py_XDECREF(predict);
	    Py_XDECREF(close_predict);
	    Py_DECREF(pModule);

	    Py_Finalize();

	}

	int main(int argc, char** argv) {
		Mat image, image1;
		image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
		if (argv[2])
			image1 = imread(argv[2], CV_LOAD_IMAGE_COLOR);	
		if (!image.data) {
			cout <<  "Could not open or find the image" << std::endl ;
	        return -1;
		}

		init();

		auto bt = chrono::high_resolution_clock::now();


		printf("%d\n", detect(image));

		if (argv[2])
			printf("%d\n", detect(image1));

    	printf("Run in %.2fms \n", chrono::duration<double, milli> (chrono::high_resolution_clock::now() - bt).count());

    	finish();
		return 0;
	}



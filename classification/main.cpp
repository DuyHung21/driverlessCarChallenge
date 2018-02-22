#include <Python.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/core/types_c.h"
#include "opencv2/opencv_modules.hpp"
#include <iostream>
#include <numpy/ndarrayobject.h>
#include "pyboost_cv3_converter.cpp"
	
	using namespace pbcvt;
	using namespace cv;
	using namespace std;

	int predict(Mat m) {
		PyObject *pName, *pModule, *pDict, *init_predict, *close_predict, *predict;
	    PyObject *pArgs, *pValue;
	    char* scriptName = "predictTS";

	    //Init
	    Py_Initialize();

	    //Get python script name
	    pName = PyUnicode_DecodeFSDefault(scriptName);


	    //Import python
	    pModule = PyImport_Import(pName);
	    Py_DECREF(pName);

	    int result;

	    if (pModule != NULL) {
	    	init_predict = PyObject_GetAttrString(pModule, "init");
	    	predict = PyObject_GetAttrString(pModule, "predict");
	    	close_predict = PyObject_GetAttrString(pModule, "finalize");

	    	if (!(init_predict && PyCallable_Check(init_predict))) {
	    		if (PyErr_Occurred())
	                PyErr_Print();
	            fprintf(stderr, "Cannot find function \"%s\"\n", "init");
	            result = -1;
	    	} else if (!(predict && PyCallable_Check(predict))) {
	    		if (PyErr_Occurred())
	                PyErr_Print();
	            fprintf(stderr, "Cannot find function \"%s\"\n", "predict");
	            result = -1;
	    	} else if (!(close_predict && PyCallable_Check(close_predict))) {
	    		if (PyErr_Occurred())
	                PyErr_Print();
	            fprintf(stderr, "Cannot find function \"%s\"\n", "finalize");
	            result = -1;
	    	} else {
	    		PyObject_CallObject(init_predict, NULL);
	    		Py_DECREF(init_predict);
	    		pArgs = PyTuple_New(1);
	    		pValue = fromMatToNDArray(m);
	    		if (!pValue) {
	    			Py_DECREF(predict);
	    			Py_DECREF(close_predict);
	    			Py_DECREF(pModule);
	    			return -1;
	    		}
	    		PyTuple_SetItem(pArgs, 0, pValue);
	    		pValue = PyObject_CallObject(predict, pArgs);
	    		printf("Done!\n");
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

	    Py_Finalize();
	    return result;
	}	

	int main(int argc, char** argv) {
		Mat image;
		image = imread(argv[1], CV_LOAD_IMAGE_COLOR);	
		if (!image.data) {
			cout <<  "Could not open or find the image" << std::endl ;
	        return -1;
		}


		printf("%d\n", predict(image));


		return 0;
	}


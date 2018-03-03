#include "TSDetector.h"


TSDetector::TSDetector(string python_path=".") {
	char* scriptName = "predictTS";
	cout<<"Hellloooo"<<endl;
	cout<<python_path<<endl;
	//Init
	Py_Initialize();

	//Import python path
	PyRun_SimpleString("import sys");
	PyRun_SimpleString(("sys.path.append(\"" + python_path + "\")").c_str());

	    //Get python script name
	pName = PyUnicode_DecodeFSDefault(scriptName);


	    //Import python
	pModule = PyImport_Import(pName);
	Py_DECREF(pName);


	if (pModule != NULL) {
	   	init_predict = PyObject_GetAttrString(pModule, "init");
	   	predict = PyObject_GetAttrString(pModule, "detect");
	    close_predict = PyObject_GetAttrString(pModule, "finalize");

	    pArgs = PyTuple_New(1);
	    pValue = PyUnicode_FromString(python_path.c_str());
	    PyTuple_SetItem(pArgs, 0, pValue);

	    PyObject_CallObject(init_predict, pArgs);
	    Py_DECREF(init_predict);
	}
}

TSDetector::~TSDetector() {
		if (!(close_predict && PyCallable_Check(close_predict))) {
	    		if (PyErr_Occurred())
	                PyErr_Print();
	            fprintf(stderr, "Cannot find function \"%s\"\n", "finalize");
	    }
		else {
			PyObject_CallObject(close_predict, NULL);
		}			                

		Py_XDECREF(pValue);
		Py_XDECREF(pArgs);
	    Py_XDECREF(init_predict);
	    Py_XDECREF(predict);
	    Py_XDECREF(close_predict);
	    Py_DECREF(pModule);

	    Py_Finalize();

}

int TSDetector::detect(const Mat& m) {
			int result;
	    if (pModule != NULL) {
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
	                Py_DECREF(pArgs);
	            } else {
	            	Py_DECREF(pValue);
	            	Py_DECREF(pArgs);
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
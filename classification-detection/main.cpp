#include <chrono>
#include "TSDetector.cpp"

	int main(int argc, char** argv) {
		Mat image, image1;
		TSDetector *detector = new TSDetector(argc==4?argv[3]:".");
		image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
		if (argv[2])
			image1 = imread(argv[2], CV_LOAD_IMAGE_COLOR);	
		if (!image.data) {
			cout <<  "Could not open or find the image" << std::endl ;
	        return -1;
		}


		auto bt = chrono::high_resolution_clock::now();


		printf("%d\n", detector->detect(image));

		if (argv[2])
			printf("%d\n", detector->detect(image1));

  		printf("Run in %.2fms \n", chrono::duration<double, milli> (chrono::high_resolution_clock::now() - bt).count());

  //   	delete detector;
		return 0;
	}



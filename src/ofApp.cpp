#include "ofApp.h"

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetVerticalSync(true);
	ofEnableDepthTest();
	ofBackground(30, 30, 30);

	// Initialize Model
	std::string modelPath = ofToDataPath("vitb_metric_indoor.onnx", true);
	try {
		depthEstimator = std::make_unique<DepthAnything>(modelPath, true);
	} catch (const std::exception & e) {
		ofLogError() << "Failed to load model: " << e.what();
		return;
	}

	// Initialize Camera
	int cameraID = 0;
#ifdef _WIN32
	cap.open(cameraID, cv::CAP_DSHOW);
#else
	cap.open(cameraID, cv::CAP_V4L2);
#endif

	if (!cap.isOpened()) cap.open(cameraID, cv::CAP_ANY);
	if (!cap.isOpened()) {
		ofLogError() << "Could not open camera.";
		return;
	}

	cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	cap.set(cv::CAP_PROP_FPS, 30);

	// Setup Threading
	running = true;

	captureThread = std::thread([this]() {
		while (running) {
			auto framePtr = std::make_shared<cv::Mat>();
			if (!cap.read(*framePtr)) {
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
				continue;
			}
			if (frameQueue.size() > 3) frameQueue.clear_except_latest(2);
			frameQueue.enqueue(framePtr);
		}
		frameQueue.set_finished();
	});

	processingThread = std::thread([this]() {
		int modelResolution = 256;
		cv::Mat resizedBuffer(modelResolution, modelResolution, CV_8UC3);

		while (running) {
			auto optFrame = frameQueue.dequeue_latest();

			if (!optFrame.has_value()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
				continue;
			}

			auto framePtr = std::move(optFrame.value());
			cv::resize(*framePtr, resizedBuffer, cv::Size(modelResolution, modelResolution));

			try {
				cv::Mat rawDepth = depthEstimator->predict(resizedBuffer);
				auto result = std::make_shared<DepthResult>();
				result->rawDepth = rawDepth.clone();
				cv::resize(*framePtr, result->colorFrame, rawDepth.size());
				resultQueue.enqueue(result);
			} catch (const std::exception & e) {
				std::cerr << "Inference error: " << e.what() << std::endl;
			}
		}
		resultQueue.set_finished();
	});

	// Setup Mesh
	pointCloudMesh.setMode(OF_PRIMITIVE_POINTS);
	glEnable(GL_POINT_SMOOTH);
	glPointSize(3);
}

//--------------------------------------------------------------
void ofApp::update() {

	auto optResult = resultQueue.dequeue_latest();

	if (optResult.has_value()) {
		auto result = std::move(optResult.value());
		cv::Mat & depthMap = result->rawDepth;
		cv::Mat & colorFrame = result->colorFrame;
		pointCloudMesh.clear();

		// Intrinsics Initial Model (TODO: Calibrate camera)
		const float cx = depthMap.cols / 2.0f;
		const float cy = depthMap.rows / 2.0f;
		const float scale_factor = depthMap.cols / 640.0f;
		const float fx = 457.0f * scale_factor;
		const float fy = 457.0f * scale_factor;

		// Point cloud
		for (int v = 0; v < depthMap.rows; ++v) {
			for (int u = 0; u < depthMap.cols; ++u) {
				float z = depthMap.at<float>(v, u);

				if (z < 0.001f || z > 3.0f) continue;

				float x = (u - cx) * z / fx;
				float y = (v - cy) * z / fy;

				pointCloudMesh.addVertex(glm::vec3(x, -y, -z));
				cv::Vec3b bgr = colorFrame.at<cv::Vec3b>(v, u);
				pointCloudMesh.addColor(ofColor(bgr[2], bgr[1], bgr[0]));
			}
		}

		// 2D Depth Image
		cv::Mat depth8U, coloredDepth;
		depthMap.convertTo(depth8U, CV_8UC1, 255.0 / 3.0);
		cv::applyColorMap(depth8U, coloredDepth, cv::COLORMAP_VIRIDIS);
		cv::cvtColor(coloredDepth, coloredDepth, cv::COLOR_BGR2RGB);
		depthTex.setFromPixels(coloredDepth.ptr(), coloredDepth.cols, coloredDepth.rows, OF_IMAGE_COLOR);
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	cam.begin();
	ofScale(100, 100, 100);

	// Center the camera
	if (pointCloudMesh.getNumVertices() > 0) {
		glm::vec3 center = pointCloudMesh.getCentroid();
		ofTranslate(-center);
	}

	pointCloudMesh.draw();

	cam.end();

	// UI Info
	ofDisableDepthTest();
	if (depthTex.isAllocated()) {
		float pxScale = 0.75;
		depthTex.draw(20, 60, 640 * pxScale, 480 * pxScale);
	}

	ofEnableDepthTest();
}

//--------------------------------------------------------------
void ofApp::exit() {
	running = false;
	frameQueue.set_finished();
	resultQueue.set_finished();

	if (captureThread.joinable()) captureThread.join();
	if (processingThread.joinable()) processingThread.join();

	cap.release();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {
	switch (key) {
	case 'F':
	case 'f':
		ofToggleFullscreen();
		break;
	}
}
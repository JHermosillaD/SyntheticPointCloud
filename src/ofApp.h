#pragma once

#include "depth_anything.hpp"
#include "ofMain.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <optional>
#include <queue>
#include <thread>

// Thread-safe queue
template <typename T>
class SafeQueue {
public:
	SafeQueue(size_t max_size = 10)
		: max_size_(max_size) { }

	bool enqueue(T value) {
		std::unique_lock<std::mutex> lock(mutex_);
		if (queue_.size() >= max_size_) {
			if (!queue_.empty()) queue_.pop();
		}
		queue_.push(std::move(value));
		cond_var_.notify_one();
		return true;
	}

	std::optional<T> dequeue() {
		std::unique_lock<std::mutex> lock(mutex_);
		while (queue_.empty() && !finished_) {
			cond_var_.wait(lock);
		}
		if (queue_.empty()) return std::nullopt;
		T result = std::move(queue_.front());
		queue_.pop();
		return result;
	}

	std::optional<T> dequeue_latest() {
		std::unique_lock<std::mutex> lock(mutex_);
		if (queue_.empty()) return std::nullopt;
		T result = std::move(queue_.back());
		std::queue<T> empty;
		std::swap(queue_, empty);
		return result;
	}

	void clear_except_latest(size_t keep_count) {
		std::unique_lock<std::mutex> lock(mutex_);
		if (queue_.size() <= keep_count) return;
		size_t to_remove = queue_.size() - keep_count;
		for (size_t i = 0; i < to_remove; i++)
			queue_.pop();
	}

	void set_finished() {
		std::unique_lock<std::mutex> lock(mutex_);
		finished_ = true;
		cond_var_.notify_all();
	}

	void clear() {
		std::unique_lock<std::mutex> lock(mutex_);
		std::queue<T> empty;
		std::swap(queue_, empty);
	}

	size_t size() {
		std::unique_lock<std::mutex> lock(mutex_);
		return queue_.size();
	}

private:
	std::queue<T> queue_;
	std::mutex mutex_;
	std::condition_variable cond_var_;
	size_t max_size_;
	bool finished_ = false;
};

struct DepthResult {
	cv::Mat rawDepth;
	cv::Mat colorFrame;
};

class ofApp : public ofBaseApp {
public:
	void setup() override;
	void update() override;
	void draw() override;
	void exit() override;
	void keyPressed(int key);

	// Viewport
	ofEasyCam cam;
	ofMesh pointCloudMesh;
	ofImage depthTex;

	// ML
	std::unique_ptr<DepthAnything> depthEstimator;
	cv::VideoCapture cap;

	// Threading
	std::thread captureThread;
	std::thread processingThread;
	std::atomic<bool> running;

	SafeQueue<std::shared_ptr<cv::Mat>> frameQueue { 5 };
	SafeQueue<std::shared_ptr<DepthResult>> resultQueue { 3 };
};
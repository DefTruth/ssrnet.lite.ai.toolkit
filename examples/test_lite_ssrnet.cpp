//
// Created by DefTruth on 2021/6/26.
//

#include "lite/lite.h"

static void test_default()
{
  std::string onnx_path = "../hub/onnx/cv/ssrnet.onnx";
  std::string test_img_path = "../resources/1.png";
  std::string save_img_path = "../logs/1.jpg";

  auto *ssrnet = new lite::cv::face::attr::SSRNet(onnx_path);

  lite::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ssrnet->detect(img_bgr, age);

  lite::utils::draw_age_inplace(img_bgr, age);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Default Version Done! Detected SSRNet Age: " << age.age << std::endl;

  delete ssrnet;
}

static void test_onnxruntime()
{
#ifdef ENABLE_ONNXRUNTIME
  std::string onnx_path = "../hub/onnx/cv/ssrnet.onnx";
  std::string test_img_path = "../resources/2.png";
  std::string save_img_path = "../logs/2_onnx.jpg";

  auto *ssrnet = new lite::onnxruntime::cv::face::attr::SSRNet(onnx_path);

  lite::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ssrnet->detect(img_bgr, age);

  lite::utils::draw_age_inplace(img_bgr, age);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "ONNXRuntime Version Done! Detected SSRNet Age: " << age.age << std::endl;

  delete ssrnet;
#endif
}

static void test_mnn()
{
#ifdef ENABLE_MNN
  std::string mnn_path = "../hub/mnn/cv/ssrnet.mnn";
  std::string test_img_path = "../resources/3.png";
  std::string save_img_path = "../logs/3_mnn.jpg";

  auto *ssrnet = new lite::mnn::cv::face::attr::SSRNet(mnn_path);

  lite::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ssrnet->detect(img_bgr, age);

  lite::utils::draw_age_inplace(img_bgr, age);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "MNN Version Done! Detected SSRNet Age: " << age.age << std::endl;

  delete ssrnet;
#endif
}

static void test_ncnn()
{
#ifdef ENABLE_NCNN
#endif
}

static void test_tnn()
{
#ifdef ENABLE_TNN
  std::string proto_path = "../hub/tnn/cv/ssrnet.opt.tnnproto";
  std::string model_path = "../hub/tnn/cv/ssrnet.opt.tnnmodel";
  std::string test_img_path = "../resources/4.png";
  std::string save_img_path = "../logs/4_tnn.jpg";

  auto *ssrnet = new lite::tnn::cv::face::attr::SSRNet(proto_path, model_path);

  lite::types::Age age;
  cv::Mat img_bgr = cv::imread(test_img_path);
  ssrnet->detect(img_bgr, age);

  lite::utils::draw_age_inplace(img_bgr, age);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "TNN Version Done! Detected SSRNet Age: " << age.age << std::endl;

  delete ssrnet;
#endif
}

static void test_lite()
{
  test_default();
  test_onnxruntime();
  test_mnn();
  test_ncnn();
  test_tnn();
}

int main(__unused int argc, __unused char *argv[])
{
  test_lite();
  return 0;
}

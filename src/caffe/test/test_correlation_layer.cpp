#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/correlation_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CorrelationLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  CorrelationLayerTest()
      : blob_bottom_0_(new Blob<Dtype>(2, 3, 5, 6)),
        blob_bottom_1_(new Blob<Dtype>(2, 3, 3, 2)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    shared_ptr<GaussianFiller<Dtype> > filler;
    FillerParameter filler_param;
    filler.reset(new GaussianFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_0_);
    filler.reset(new GaussianFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_1_);
    blob_bottom_vec_0_.push_back(blob_bottom_0_);
    blob_bottom_vec_0_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }

  virtual ~CorrelationLayerTest() {
    delete blob_bottom_0_; delete blob_bottom_1_;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_0_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(CorrelationLayerTest, TestDtypesAndDevices);

TYPED_TEST(CorrelationLayerTest, TestSetupNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CorrelationLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_vec_[0]->channels(),
      this->blob_bottom_vec_0_[1]->height()*this->blob_bottom_vec_0_[1]->width());
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_vec_0_[1]->num());
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_vec_0_[0]->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_vec_0_[0]->width());
}

TYPED_TEST(CorrelationLayerTest, TestForwardNum) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CorrelationLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_0_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_0_, this->blob_top_vec_);
  for (int n = 0; n < this->blob_bottom_vec_0_[0]->num(); ++n) {
    for (int c = 0; c < this->blob_top_->channels(); ++c) {
      int b2_col = c%this->blob_bottom_vec_0_[1]->width();
      int b2_row = c/this->blob_bottom_vec_0_[1]->width();
      for (int h = 0; h < this->blob_top_->height(); ++h) {
        for (int w = 0; w < this->blob_top_->width(); ++w) {
          Dtype corr = 0;
          for (int ic = 0;ic<this->blob_bottom_vec_0_[1]->channels();ic++){
              corr = corr+this->blob_bottom_vec_0_[0]->data_at(n, ic, h, w)*
                  this->blob_bottom_vec_0_[1]->data_at(n, ic, b2_row, b2_col);
          }
          EXPECT_GE(this->blob_top_->data_at(n, c, h, w),corr-1e-4);
          EXPECT_LE(this->blob_top_->data_at(n, c, h, w),corr+1e-4);
        }
      }
    }
  }
}

TYPED_TEST(CorrelationLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  CorrelationLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_0_,
    this->blob_top_vec_);
}
//

}  // namespace caffe

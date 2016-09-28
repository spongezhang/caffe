#include <vector>

#include "caffe/layers/correlation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CorrelationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void CorrelationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //only 2 images 
  CHECK_EQ(2, bottom.size());
  int channel_1 = bottom[0]->channels();
  int channel_2 = bottom[1]->channels();
  CHECK_EQ(channel_1,channel_2);
  num_ = bottom[0]->num();
  channel_number = channel_1;
  vector<int> top_shape = bottom[0]->shape();
  spatial_dim_1 = bottom[0]->height()*bottom[0]->width();
  spatial_dim_2 = bottom[1]->height()*bottom[1]->width();
  dim_count_1 = spatial_dim_1*channel_1;
  dim_count_2 = spatial_dim_2*channel_2;
  top_shape[1] = spatial_dim_2;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void CorrelationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* bottom_data_1 = bottom[0]->cpu_data();
  const Dtype* bottom_data_2 = bottom[1]->cpu_data();
  
  for (int n = 0; n < this->num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans,CblasNoTrans, spatial_dim_2, spatial_dim_1, channel_number,
        (Dtype)1., bottom_data_2+n*spatial_dim_2*channel_number, bottom_data_1+n*spatial_dim_1*channel_number,
        (Dtype)0., top_data + n*spatial_dim_1*spatial_dim_2);
  }
}

template <typename Dtype>
void CorrelationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff_1 = top[0]->cpu_diff();
  Dtype* bottom_diff_1 = bottom[0]->mutable_cpu_diff();
  Dtype* bottom_diff_2 = bottom[1]->mutable_cpu_diff();
  const Dtype* bottom_data_1 = bottom[0]->cpu_data();
  const Dtype* bottom_data_2 = bottom[1]->cpu_data();
  if (propagate_down[0]) {
    for (int n = 0; n < this->num_; ++n) {
      // gradient w.r.t. bottom data, if necessary.
      if (propagate_down[0]) {
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channel_number, spatial_dim_1, spatial_dim_2,
          (Dtype)1.,bottom_data_2+n*spatial_dim_2*channel_number, top_diff_1 + n*spatial_dim_1*spatial_dim_2,
          (Dtype)0., bottom_diff_1 + n*spatial_dim_1*channel_number);
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channel_number, spatial_dim_2, spatial_dim_1,
          (Dtype)1.,bottom_data_1+n*spatial_dim_1*channel_number, top_diff_1 + n*spatial_dim_2*spatial_dim_1,
          (Dtype)0., bottom_diff_2 + n*spatial_dim_2*channel_number);
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(CorrelationLayer);
#endif

INSTANTIATE_CLASS(CorrelationLayer);
REGISTER_LAYER_CLASS(Correlation);

}  // namespace caffe

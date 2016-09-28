#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/l2_normalization_layer.hpp"

namespace caffe {

template <typename Dtype>
void L2NormalizationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  channel_wise = this->layer_param_.l2_normalization_param().channel_wise();
}


template <typename Dtype>
void L2NormalizationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
     const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  if(channel_wise){
    sum_multiplier_.Reshape(1, bottom[0]->channels(), 1, 1);
    norm_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
    temp_dot_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  }
  else{
    sum_multiplier_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
    norm_.Reshape(bottom[0]->num(), bottom[0]->channels(), 1, 1);
    temp_dot_.Reshape(bottom[0]->num(),bottom[0]->channels(), 1, 1);
  }
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  
  for (int i = 0; i < sum_multiplier_.count(); ++i) {
    multiplier_data[i] = 1.;
  }
  
  square_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void L2NormalizationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
   const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* square_data = square_.mutable_cpu_data();
  Dtype* norm_data = norm_.mutable_cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int dim = bottom[0]->count() / bottom[0]->num();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  caffe_copy(bottom[0]->count(), bottom_data, square_data);
  // do the normalizition.

  for (int i = 0; i < num; ++i) {
    // squre each element
    caffe_sqr<Dtype>(dim, square_data + i * dim, square_data + i * dim);
    if(channel_wise){
      // sum cross the channel
      caffe_cpu_gemv<Dtype>(CblasTrans, channels, spatial_dim, 1.,
          square_data + i * dim, sum_multiplier_.cpu_data(), 0.,
          norm_data + i * spatial_dim);
      // root the square norm_data
      caffe_powx<Dtype>(spatial_dim, norm_data + i * spatial_dim, 0.5,
          norm_data + i * spatial_dim);
      // division
      for (int j = 0; j < channels; j++) {
        caffe_div(spatial_dim, top_data + top[0]->offset(i, j), 
            norm_data + i * spatial_dim, top_data + top[0]->offset(i, j));
      }
    }
    else{
      // sum cross the channel
      caffe_cpu_gemv<Dtype>(CblasNoTrans, channels, spatial_dim, 1.,
              square_data + i * dim, sum_multiplier_.cpu_data(), 0.,
              norm_data + i * channels);
      // root the square norm_data
      caffe_powx<Dtype>(channels, norm_data + i * channels, 0.5,
          norm_data + i * channels);
      // division
      for (int j = 0; j < channels; j++) {
        caffe_scal(spatial_dim, (Dtype)1.0/(*(norm_data + i *channels + j)),
                top_data + i*dim + j*spatial_dim);
      }
    }
  }
}

template <typename Dtype>
void L2NormalizationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
  const vector<bool>& propagate_down,
  const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* norm_data = norm_.mutable_cpu_data();
  Dtype* temp_dot_data = temp_dot_.mutable_cpu_data();
  Dtype* temp_data = square_.mutable_cpu_data();//just reuse the square_
  int num = top[0]->num();
  int channels = top[0]->channels();
  int dim = top[0]->count() / top[0]->num();
  int spatial_dim = top[0]->height() * top[0]->width();
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < num; ++i) {
    // b_diff = t_diff / norm - dot(t_diff, t_data) / (norm)^2 * bottom_data
    // compute dot(top_diff, top_data)
    if(channel_wise){
      for (int k = 0; k < spatial_dim; ++k) {
        temp_dot_data[i * spatial_dim + k] = caffe_cpu_strided_dot<Dtype>(channels,
            top_diff + i * dim + k, spatial_dim,
            top_data + i * dim + k, spatial_dim)  /  
            ( norm_data[i * spatial_dim + k] * norm_data[i * spatial_dim + k] );
      }
      // b_diff = t_diff / norm - dot(t_diff, t_data) / (norm)^2 * bottom_data
      for (int j = 0; j < channels; j++) {
        caffe_div(spatial_dim, bottom_diff + top[0]->offset(i, j), 
            norm_data + i * spatial_dim, bottom_diff + top[0]->offset(i, j));
        caffe_mul(spatial_dim, bottom_data + top[0]->offset(i, j), 
            temp_dot_data + i * spatial_dim, temp_data + top[0]->offset(i, j));
        caffe_axpy(spatial_dim, Dtype(-1.0), temp_data + top[0]->offset(i, j), 
            bottom_diff + top[0]->offset(i, j));
      }
    }
    else{
      for (int k = 0; k < channels; ++k) {
        temp_dot_data[i * channels + k] = caffe_cpu_strided_dot<Dtype>(spatial_dim,
            top_diff + i * dim + k*spatial_dim, 1,
            top_data + i * dim + k*spatial_dim, 1)  /  
            ( norm_data[i * channels + k] * norm_data[i * channels + k] );
      }
      // b_diff = t_diff / norm - dot(t_diff, t_data) / (norm)^2 * bottom_data
      for (int j = 0; j < channels; j++) {
        caffe_scal(spatial_dim, (Dtype)1.0/(*(norm_data + i *channels + j)),
                bottom_diff + i*channels*spatial_dim+j*spatial_dim);
        caffe_cpu_axpby(spatial_dim, (Dtype)(*(temp_dot_data + i*channels + j)),
                bottom_data + i*channels*spatial_dim+j*spatial_dim,
                (Dtype)0.0, temp_data + i*channels*spatial_dim+j*spatial_dim);
        caffe_axpy(spatial_dim, Dtype(-1.0), temp_data + top[0]->offset(i, j), 
            bottom_diff + top[0]->offset(i, j));
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(L2NormalizationLayer);
#endif

INSTANTIATE_CLASS(L2NormalizationLayer);
REGISTER_LAYER_CLASS(L2Normalization);

} //namespace caffe

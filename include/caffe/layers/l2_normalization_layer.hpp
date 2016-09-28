#ifndef CAFFE_L2_NORMALIZATION_LAYER_HPP_
#define CAFFE_L2_NORMALIZATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Computes the multinomial logistic loss for a one-of-many
 *        classification task, passing real-valued predictions through a
 *        softmax to get a probability distribution over classes.
 *
 * This layer should be preferred over separate
 * SoftmaxLayer + MultinomialLogisticLossLayer
 * as its gradient computation is more numerically stable.
 * At test time, this layer can be replaced simply by a SoftmaxLayer.
 *
 * @param bottom input Blob vector (length 2)
 *   -# @f$ (N \times C \times H \times W) @f$
 *      the predictions @f$ x @f$, a Blob with values in
 *      @f$ [-\infty, +\infty] @f$ indicating the predicted score for each of
 *      the @f$ K = CHW @f$ classes. This layer maps these scores to a
 *      probability distribution over classes using the softmax function
 *      @f$ \hat{p}_{nk} = \exp(x_{nk}) /
 *      \left[\sum_{k'} \exp(x_{nk'})\right] @f$ (see SoftmaxLayer).
 *   -# @f$ (N \times 1 \times 1 \times 1) @f$
 *      the labels @f$ l @f$, an integer-valued Blob with values
 *      @f$ l_n \in [0, 1, 2, ..., K - 1] @f$
 *      indicating the correct class label among the @f$ K @f$ classes
 * @param top output Blob vector (length 1)
 *   -# @f$ (1 \times 1 \times 1 \times 1) @f$
 *      the computed cross-entropy classification loss: @f$ E =
 *        \frac{-1}{N} \sum\limits_{n=1}^N \log(\hat{p}_{n,l_n})
 *      @f$, for softmax output class probabilites @f$ \hat{p} @f$
 */
template <typename Dtype>
class L2NormalizationLayer : public Layer<Dtype> {
public:
  explicit L2NormalizationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "L2Normalization"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  bool channel_wise;
  /// sum_multiplier is used to carry out sum using BLAS 1 x ch x 1 x 1
  Blob<Dtype> sum_multiplier_;
  /// square result n X ch x h x w
  Blob<Dtype> square_;
  /// norm is an intermediate Blob to hold temporary results. n * 1 * h * w
  Blob<Dtype> norm_;
  /// temp_dot n * 1 * h * w
  Blob<Dtype> temp_dot_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_

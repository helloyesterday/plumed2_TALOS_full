#ifdef __PLUMED_HAS_DYNET

#ifndef DYNET_TOOLS_H
#define DYNET_TOOLS_H

/**
 * \file rnnlm-batch.h
 * \defgroup ffbuilders ffbuilders
 * \brief Feed forward nets builders
 *
 * An example implementation of a simple multilayer perceptron
 *
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <dynet/nodes.h>
#include <dynet/dynet.h>
#include <dynet/training.h>
#include <dynet/timing.h>
#include <dynet/expr.h>
#include <dynet/io.h>

namespace dytools {

using namespace dynet;

Trainer* new_traniner(const std::string& algorithm,ParameterCollection& pc,std::string& fullname);
Trainer* new_traniner(const std::string& algorithm,ParameterCollection& pc,const std::vector<float>& params,std::string& fullname);

/**
 * \ingroup ffbuilders
 * Common activation functions used in multilayer perceptrons
 */
enum Activation {
  SIGMOID, /**< `SIGMOID` : Sigmoid function \f$x\longrightarrow \frac {1} {1+e^{-x}}\f$ */
  TANH, /**< `TANH` : Tanh function \f$x\longrightarrow \frac {1-e^{-2x}} {1+e^{-2x}}\f$ */
  RELU, /**< `RELU` : Rectified linear unit \f$x\longrightarrow \max(0,x)\f$ */
  LINEAR, /**< `LINEAR` : Identity function \f$x\longrightarrow x\f$ */
  SOFTMAX /**< `SOFTMAX` : Softmax function \f$\textbf{x}=(x_i)_{i=1,\dots,n}\longrightarrow \frac {e^{x_i}}{\sum_{j=1}^n e^{x_j} })_{i=1,\dots,n}\f$ */
};

Activation activation_function(const std::string& a);

/**
 * \ingroup ffbuilders
 * \struct Layer
 * \brief Simple layer structure
 * \details Contains all parameters defining a layer
 *
 */
struct Layer {
public:
  unsigned input_dim; /**< Input dimension */
  unsigned output_dim; /**< Output dimension */
  Activation activation = LINEAR; /**< Activation function */
  float dropout_rate = 0; /**< Dropout rate */
  /**
   * \brief Build a feed forward layer
   *
   * \param input_dim Input dimension
   * \param output_dim Output dimension
   * \param activation Activation function
   * \param dropout_rate Dropout rate
   */
  Layer(unsigned input_dim, unsigned output_dim, Activation activation, float dropout_rate) :
    input_dim(input_dim),
    output_dim(output_dim),
    activation(activation),
    dropout_rate(dropout_rate) {};
  Layer() {};
};

/**
 * \ingroup ffbuilders
 * \struct MLP
 * \brief Simple multilayer perceptron
 *
 */

/**
 * \ingroup ffbuilders
 * \struct MLP
 * \brief Simple multilayer perceptron
 *
 */
struct MLP {
protected:
  // Hyper-parameters
  unsigned LAYERS = 0;
  unsigned params_num = 0;
  unsigned input_dim; /**< Input dimension */
  unsigned output_dim; /**< Output dimension */

  // Layers
  std::vector<Layer> layers;
  // Parameters
  std::vector<std::vector<Parameter>> params;
  std::vector<std::vector<unsigned>> params_size;

  bool dropout_active = true;

public:
  unsigned get_layers() const {return LAYERS;}
  unsigned get_output_dim() const {return input_dim;}
  unsigned get_input_dim() const {return output_dim;}
  void clip(float left,float right,bool clip_last_layer=false);
  void clip_inplace(float left,float right,bool clip_last_layer=false);
  
  unsigned parameters_number() const {return params_num;}
  
   /**
   * \brief Default constructor
   * \details Dont forget to add layers!
   */
  explicit MLP():LAYERS(0) {}
  
   /**
   * \brief Default constructor
   * \details Dont forget to add layers!
   */
  explicit MLP(ParameterCollection & model):LAYERS(0){}
  
  /**
   * \brief Returns a Multilayer perceptron
   * \details Creates a feedforward multilayer perceptron based on a list of layer descriptions
   *
   * \param model ParameterCollection to contain parameters
   * \param layers Layers description
   */
  explicit MLP(ParameterCollection& model,std::vector<Layer> layers);
  
  /**
   * \brief Append a layer at the end of the network
   * \details [long description]
   *
   * \param model [description]
   * \param layer [description]
   */
  void append(ParameterCollection& model, Layer layer);
  
    /**
   * \brief Run the MLP on an input vector/batch
   *
   * \param x Input expression (vector or batch)
   * \param cg Computation graph
   *
   * \return [description]
   */
  Expression run(Expression x,ComputationGraph& cg);
                 
  /**
   * \brief Run the MLP on an input vector/batch
   *
   * \param x Input expression (vector or batch)
   * \param cg Computation graph
   *
   * \return [description]
   */
  Expression get_grad(Expression x,ComputationGraph& cg);
  
  /**
   * \brief Return the negative log likelihood for the (batched) pair (x,y)
   * \details For a batched input \f$\{x_i\}_{i=1,\dots,N}\f$, \f$\{y_i\}_{i=1,\dots,N}\f$, this computes \f$\sum_{i=1}^N \log(P(y_i\vert x_i))\f$ where \f$P(\textbf{y}\vert x_i)\f$ is modelled with $\mathrm{softmax}(MLP(x_i))$
   *
   * \param x Input batch
   * \param labels Output labels
   * \param cg Computation graph
   * \return Expression for the negative log likelihood on the batch
   */
  Expression get_nll(Expression x,std::vector<unsigned> labels,ComputationGraph& cg);
  
  /**
   * \brief Predict the most probable label
   * \details Returns the argmax of the softmax of the networks output
   *
   * \param x Input
   * \param cg Computation graph
   *
   * \return Label index
   */
  int predict(Expression x,ComputationGraph& cg);
  
    /**
   * \brief Enable dropout
   * \details This is supposed to be used during training or during testing if you want to sample outputs using montecarlo
   */
  void enable_dropout() {
    dropout_active = true;
  }

  /**
   * \brief Disable dropout
   * \details Do this during testing if you want a deterministic network
   */
  void disable_dropout() {
    dropout_active = false;
  }

  /**
   * \brief Check wether dropout is enabled or not
   *
   * \return Dropout state
   */
  bool is_dropout_enabled() {
    return dropout_active;
  }
  
  void set_parameters(const std::vector<float>&);
  std::vector<float> get_parameters();

private:
  inline Expression activate(Expression h, Activation f);
  inline Expression activate_grad(Expression h, Activation f);
};

class WGAN {
public:
	WGAN(MLP& _nn,unsigned _bsize,unsigned _ntarget,
	std::vector<dynet::real>& x_svalues,
	std::vector<dynet::real>& x_tvalues,
	std::vector<dynet::real>& p_target);
	WGAN(MLP& _nn,unsigned _bsize,unsigned _ntarget,
	std::vector<dynet::real>& x_svalues,
	std::vector<dynet::real>& x_tvalues);
	
	void clear_cg(){cg.clear();}

	unsigned batch_size() const {return bsize;}
	void set_batch_size(unsigned new_size) {bsize=new_size;}
	float update(Trainer& trainer);
private:
	ComputationGraph cg;
	MLP nn;
	unsigned bsize;
	unsigned ntarget;
	
	float clip_min;
	float clip_max;
	
	Expression x_sample;
	Expression x_target;
	Expression y_sample;
	Expression y_target;
	Expression loss_expr;
	
	void set_expression(std::vector<dynet::real>& x_svalues,
		std::vector<dynet::real>& x_tvalues,
		std::vector<dynet::real>& p_target);
	void set_expression(std::vector<dynet::real>& x_svalues,
		std::vector<dynet::real>& x_tvalues);
};


}

#endif

#endif

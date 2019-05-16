#ifdef __PLUMED_HAS_DYNET

/**
 * \file rnnlm-batch.h
 * \defgroup ffbuilders ffbuilders
 * \brief Feed forward nets builders
 *
 * An example implementation of a simple multilayer perceptron
 *
 */

#include "DynetTools.h"

namespace dytools {

using namespace dynet;

Activation activation_function(const std::string& a)
{
	if(a=="SIGMOID"||a=="sigmoid"||a=="Sigmoid")
		return Activation::SIGMOID;
	if(a=="TANH"||a=="tanh"||a=="Tanh")
		return Activation::TANH;
	if(a=="RELU"||a=="relu"||a=="Relu"||a=="ReLU")
		return Activation::RELU;
	if(a=="LINEAR"||a=="linear"||a=="Linear")
		return Activation::LINEAR;
	if(a=="SOFTMAX"||a=="softmax"||a=="Softmax"||a=="SoftMax"||a=="SoftMAX")
		return Activation::SOFTMAX;
	std::cerr<<"ERROR! Can't recognize the activation function "+a<<std::endl;
	exit(-1);
}

  /**
   * \brief Returns a Multilayer perceptron
   * \details Creates a feedforward multilayer perceptron based on a list of layer descriptions
   *
   * \param model ParameterCollection to contain parameters
   * \param layers Layers description
   */
MLP::MLP(ParameterCollection& model,std::vector<Layer> layers)
{
    // Verify layers compatibility
    for (unsigned l = 0; l < layers.size() - 1; ++l) {
      if (layers[l].output_dim != layers[l + 1].input_dim)
        throw std::invalid_argument("Layer dimensions don't match");
    }

    // Register parameters in model
    for (Layer layer : layers) {
      append(model, layer);
    }
}

  /**
   * \brief Append a layer at the end of the network
   * \details [long description]
   *
   * \param model [description]
   * \param layer [description]
   */
void MLP::append(ParameterCollection& model, Layer layer)
{
    // Check compatibility
    if (LAYERS > 0)
    {
      if (layers[LAYERS - 1].output_dim != layer.input_dim)
        throw std::invalid_argument("Layer dimensions don't match");
      output_dim=layer.output_dim;
	}
	else
      input_dim=layer.input_dim;

    // Add to layers
    layers.push_back(layer);
    LAYERS++;
    // Register parameters
    Parameter W = model.add_parameters({layer.output_dim, layer.input_dim});
    Parameter b = model.add_parameters({layer.output_dim});
    params.push_back({W, b});
    unsigned nw = layer.output_dim * layer.input_dim;
    unsigned nb = layer.output_dim;
    params_size.push_back({nw,nb});
    params_num+=nw;
    params_num+=nb;
}

  /**
   * \brief Run the MLP on an input vector/batch
   *
   * \param x Input expression (vector or batch)
   * \param cg Computation graph
   *
   * \return [description]
   */
Expression MLP::run(Expression x,ComputationGraph& cg)
{
    // Expression for the current hidden state
    Expression h_cur = x;
    for (unsigned l = 0; l < LAYERS; ++l) {
      // Initialize parameters in computation graph
      Expression W = parameter(cg, params[l][0]);
      Expression b = parameter(cg, params[l][1]);
      // Aplly affine transform
      Expression a = affine_transform({b, W, h_cur});
      // Apply activation function
      Expression h = activate(a, layers[l].activation);
      // Take care of dropout
      Expression h_dropped;
      if (layers[l].dropout_rate > 0) {
        if (dropout_active) {
          // During training, drop random units
          Expression mask = random_bernoulli(cg, {layers[l].output_dim}, 1 - layers[l].dropout_rate);
          h_dropped = cmult(h, mask);
        } else {
          // At test time, multiply by the retention rate to scale
          h_dropped = h * (1 - layers[l].dropout_rate);
        }
      } else {
        // If there's no dropout, don't do anything
        h_dropped = h;
      }
      // Set current hidden state
      h_cur = h_dropped;
    }

    return h_cur;
}

  /**
   * \brief Run the MLP on an input vector/batch
   *
   * \param x Input expression (vector or batch)
   * \param cg Computation graph
   *
   * \return [description]
   */
Expression MLP::get_grad(Expression x,ComputationGraph& cg)
{
    // Expression for the current hidden state
    Expression h_cur = x;
    std::vector<Expression> a_vec;
    for (unsigned l = 0; l < LAYERS; ++l) {
      // Initialize parameters in computation graph
      Expression W = parameter(cg, params[l][0]);
      Expression b = parameter(cg, params[l][1]);
      // Aplly affine transform
      Expression a = affine_transform({b, W, h_cur});
      
      a_vec.push_back(a);
      
      // Apply activation function
      Expression h = activate(a, layers[l].activation);
      // Take care of dropout
      Expression h_dropped;
      if (layers[l].dropout_rate > 0) {
        if (dropout_active) {
          // During training, drop random units
          Expression mask = random_bernoulli(cg, {layers[l].output_dim}, 1 - layers[l].dropout_rate);
          h_dropped = cmult(h, mask);
        } else {
          // At test time, multiply by the retention rate to scale
          h_dropped = h * (1 - layers[l].dropout_rate);
        }
      } else {
        // If there's no dropout, don't do anything
        h_dropped = h;
      }
      // Set current hidden state
      h_cur = h_dropped;
    }
    
    Expression y_grad;
    for (unsigned l = 0; l < LAYERS; ++l) {
      unsigned id=LAYERS-1-l;
      std::cout<<id<<std::endl;
      Expression W = parameter(cg, params[id][0]);
      std::cout<<W.dim()<<std::endl;
      if(l==0)
      {
        if(layers[id].activation==LINEAR)
		  y_grad = W;
		else
		{
          Expression h_grad=activate_grad(a_vec[id],layers[id].activation);
          y_grad = h_grad * W;
		}
	  }
      else
      {
        Expression g_cur=y_grad;
        if(layers[id].activation!=LINEAR)
        {
	      Expression h_grad=activate_grad(a_vec[id],layers[id].activation);
          g_cur = y_grad * h_grad;
        }
        y_grad = g_cur * W;
	  }
    }
    Expression tt=transpose(y_grad);
	std::cout<<tt.dim()<<std::endl;
	
    return y_grad;
}
  
  /**
   * \brief Return the negative log likelihood for the (batched) pair (x,y)
   * \details For a batched input \f$\{x_i\}_{i=1,\dots,N}\f$, \f$\{y_i\}_{i=1,\dots,N}\f$, this computes \f$\sum_{i=1}^N \log(P(y_i\vert x_i))\f$ where \f$P(\textbf{y}\vert x_i)\f$ is modelled with $\mathrm{softmax}(MLP(x_i))$
   *
   * \param x Input batch
   * \param labels Output labels
   * \param cg Computation graph
   * \return Expression for the negative log likelihood on the batch
   */
Expression MLP::get_nll(Expression x,std::vector<unsigned> labels,ComputationGraph& cg)
{
    // compute output
    Expression y = run(x, cg);
    // Do softmax
    Expression losses = pickneglogsoftmax(y, labels);
    // Sum across batches
    return sum_batches(losses);
}

  /**
   * \brief Predict the most probable label
   * \details Returns the argmax of the softmax of the networks output
   *
   * \param x Input
   * \param cg Computation graph
   *
   * \return Label index
   */
int MLP::predict(Expression x,ComputationGraph& cg)
{
    // run MLP to get class distribution
    Expression y = run(x, cg);
    // Get values
    std::vector<float> probs = as_vector(cg.forward(y));
    // Get argmax
    unsigned argmax = 0;
    for (unsigned i = 1; i < probs.size(); ++i) {
      if (probs[i] > probs[argmax])
        argmax = i;
    }

    return argmax;
}

void MLP::clip(float left,float right,bool clip_last_layer)
{
	for(unsigned i=0;i!=params.size();++i)
	{
		if((i+1)!=params.size()||clip_last_layer)
		{
			for(unsigned j=0;j!=params[i].size();++j)
				params[i][j].get_storage().clip(left,right);
		}
	}
}

void MLP::clip_inplace(float left,float right,bool clip_last_layer)
{
	for(unsigned i=0;i!=params.size();++i)
	{
		if((i+1)!=params.size()||clip_last_layer)
		{
			for(unsigned j=0;j!=params[i].size();++j)
				params[i][j].clip_inplace(left,right);
		}
	}
}

std::vector<float> MLP::get_parameters()
{
	std::vector<float> param_values;
	for(unsigned i=0;i!=params.size();++i)
	{
		for(unsigned j=0;j!=params[i].size();++j)
		{
			std::vector<float> vv=as_vector(*params[i][j].values());
			param_values.insert(param_values.end(),vv.begin(),vv.end());
		}
	}
	return param_values;
}

void MLP::set_parameters(const std::vector<float>& param_values)
{
	if(param_values.size()<params_num)
	{
		std::cerr<<"ERROR! The number of the parameter overflow!"<<std::endl;
		exit(-1);
	}
	unsigned ival=0;
	for(unsigned i=0;i!=params.size();++i)
	{
		for(unsigned j=0;j!=params[i].size();++j)
		{
			std::vector<float> new_params;
			for(unsigned k=0;k!=params_size[i][j];++k)
				new_params.push_back(param_values[ival++]);
			params[i][j].set_value(new_params);
		}
	}
}

inline Expression MLP::activate(Expression h, Activation f)
{
    switch (f) {
    case LINEAR:
      return h;
      break;
    case RELU:
      return rectify(h);
      break;
    case SIGMOID:
      return logistic(h);
      break;
    case TANH:
      return tanh(h);
      break;
    case SOFTMAX:
      return softmax(h);
      break;
    default:
      throw std::invalid_argument("Unknown activation function");
      break;
    }
}

inline Expression MLP::activate_grad(Expression h, Activation f)
{
	Expression a;
    switch (f) {
    case LINEAR:
      return ones(*(h.pg),h.dim());
      break;
    case RELU:
      a=rectify(h);
      return cdiv(a,a+std::numeric_limits<float>::epsilon());
      break;
    case SIGMOID:
      a=logistic(h);
      return cmult(a,1.0-a);
      break;
    case TANH:
      a=tanh(h);
      return 1.0-cmult(a,a);
      break;
    case SOFTMAX:
      a=softmax(h);
      return a*transpose(a);
      break;
    default:
      throw std::invalid_argument("Unknown activation function");
      break;
    }
  }

WGAN::WGAN(MLP& _nn,unsigned _bsize,unsigned _ntarget,
	std::vector<dynet::real>& x_svalues,
	std::vector<dynet::real>& x_tvalues,
	std::vector<dynet::real>& p_target):
nn(_nn),
bsize(_bsize),
ntarget(_ntarget),
clip_min(-0.01),
clip_max(0.01)
{
	if(nn.get_output_dim()!=1)
	{
		std::cerr<<"ERROR! the output dimension must be one!"<<std::endl;
		std::exit(-1);
	}
	set_expression(x_svalues,x_tvalues,p_target);
}

WGAN::WGAN(MLP& _nn,unsigned _bsize,unsigned _ntarget,
	std::vector<dynet::real>& x_svalues,
	std::vector<dynet::real>& x_tvalues):
nn(_nn),
bsize(_bsize),
ntarget(_ntarget),
clip_min(-0.01),
clip_max(0.01)
{
	if(nn.get_output_dim()!=1)
	{
		std::cerr<<"ERROR! the output dimension must be one!"<<std::endl;
		std::exit(-1);
	}
	set_expression(x_svalues,x_tvalues);
}

void WGAN::set_expression(std::vector<dynet::real>& x_svalues,
	std::vector<dynet::real>& x_tvalues,
	std::vector<dynet::real>& p_target)
{
	Dim xs_dim({nn.get_input_dim()},bsize);
	Dim xt_dim({nn.get_input_dim()},ntarget);
	Dim p_dim({1},ntarget);
	
	x_sample=input(cg,xs_dim,&x_svalues);
	x_target=input(cg,xt_dim,&x_tvalues);
	Expression target_weights=input(cg,p_dim,&p_target);
	
	y_sample=nn.run(x_sample,cg);
	y_target=nn.run(x_target,cg);
	
	Expression loss_sample=mean_elems(y_sample);
	// the target target distribution must be normalized!
	Expression loss_target=dot_product(y_target,target_weights);
	loss_expr=loss_sample-loss_target;
}

void WGAN::set_expression(std::vector<dynet::real>& x_svalues,
	std::vector<dynet::real>& x_tvalues)
{
	Dim xs_dim({nn.get_input_dim()},bsize);
	Dim xt_dim({nn.get_input_dim()},ntarget);
	
	x_sample=input(cg,xs_dim,&x_svalues);
	x_target=input(cg,xt_dim,&x_tvalues);
	
	y_sample=nn.run(x_sample,cg);
	y_target=nn.run(x_target,cg);
	
	Expression loss_sample=mean_elems(y_sample);
	Expression loss_target=mean_elems(y_target);
	
	loss_expr=loss_sample-loss_target;
}

float WGAN::update(Trainer& trainer)
{
	float loss=as_scalar(cg.forward(loss_expr));
	cg.backward(loss_expr,true);
	trainer.update();
	return loss;
}

Trainer* new_traniner(const std::string& algorithm,ParameterCollection& pc,std::string& fullname)
{
	if(algorithm=="SimpleSGD"||algorithm=="simpleSGD"||algorithm=="simplesgd"||algorithm=="SGD"||algorithm=="sgd")
	{
		fullname="Stochastic gradient descent";
		Trainer *trainer = new SimpleSGDTrainer(pc);
		return trainer;
	}
	if(algorithm=="CyclicalSGD"||algorithm=="cyclicalSGD"||algorithm=="cyclicalsgd"||algorithm=="CSGD"||algorithm=="csgd")
	{
		fullname="Cyclical learning rate SGD";
		Trainer *trainer = new CyclicalSGDTrainer(pc);
		return trainer;
	}
	if(algorithm=="MomentumSGD"||algorithm=="momentumSGD"||algorithm=="momentumSGD"||algorithm=="MSGD"||algorithm=="msgd")
	{
		fullname="SGD with momentum";
		Trainer *trainer = new MomentumSGDTrainer(pc);
		return trainer;
	}
	if(algorithm=="Adagrad"||algorithm=="adagrad"||algorithm=="adag"||algorithm=="ADAG")
	{
		fullname="Adagrad optimizer";
		Trainer *trainer = new AdagradTrainer(pc);
		return trainer;
	}
	if(algorithm=="Adadelta"||algorithm=="adadelta"||algorithm=="AdaDelta"||algorithm=="AdaD"||algorithm=="adad"||algorithm=="ADAD")
	{
		fullname="AdaDelta optimizer";
		Trainer *trainer = new AdadeltaTrainer(pc);
		return trainer;
	}
	if(algorithm=="RMSProp"||algorithm=="rmsprop"||algorithm=="rmsp"||algorithm=="RMSP")
	{
		fullname="RMSProp optimizer";
		Trainer *trainer = new RMSPropTrainer(pc);
		return trainer;
	}
	if(algorithm=="Adam"||algorithm=="adam"||algorithm=="ADAM")
	{
		fullname="Adam optimizer";
		Trainer *trainer = new AdamTrainer(pc);
		return trainer;
	}
	if(algorithm=="AMSGrad"||algorithm=="Amsgrad"||algorithm=="Amsg"||algorithm=="amsg")
	{
		fullname="AMSGrad optimizer";
		Trainer *trainer = new AmsgradTrainer(pc);
		return trainer;
	}
	return NULL;
}

Trainer* new_traniner(const std::string& algorithm,ParameterCollection& pc,const std::vector<float>& params,std::string& fullname)
{
	if(params.size()==0)
		return new_traniner(algorithm,pc,fullname);

	if(algorithm=="SimpleSGD"||algorithm=="simpleSGD"||algorithm=="simplesgd"||algorithm=="SGD"||algorithm=="sgd")
	{
		fullname="Stochastic gradient descent";
		Trainer *trainer = new SimpleSGDTrainer(pc,params[0]);
		return trainer;
	}
	if(algorithm=="CyclicalSGD"||algorithm=="cyclicalSGD"||algorithm=="cyclicalsgd"||algorithm=="CSGD"||algorithm=="csgd")
	{
		fullname="Cyclical learning rate SGD";
		Trainer *trainer=NULL;
		if(params.size()<2)
		{
			std::cerr<<"ERROR! CyclicalSGD needs at least two learning rates"<<std::endl;
			exit(-1);
		}
		else if(params.size()==2)
			trainer = new CyclicalSGDTrainer(pc,params[0],params[1]);
		else if(params.size()==3)
			trainer = new CyclicalSGDTrainer(pc,params[0],params[1],params[2]);
		else if(params.size()==4)
			trainer = new CyclicalSGDTrainer(pc,params[0],params[1],params[2],params[3]);
		else
			trainer = new CyclicalSGDTrainer(pc,params[0],params[1],params[2],params[3],params[4]);
		return trainer;
	}
	if(algorithm=="MomentumSGD"||algorithm=="momentumSGD"||algorithm=="momentumSGD"||algorithm=="MSGD"||algorithm=="msgd")
	{
		fullname="SGD with momentum";
		Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new MomentumSGDTrainer(pc,params[0]);
		else
			trainer = new MomentumSGDTrainer(pc,params[0],params[1]);
		return trainer;
	}
	if(algorithm=="Adagrad"||algorithm=="adagrad"||algorithm=="adag"||algorithm=="ADAG")
	{
		fullname="Adagrad optimizer";
		Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new AdagradTrainer(pc,params[0]);
		else
			trainer = new AdagradTrainer(pc,params[0],params[1]);
		return trainer;
	}
	if(algorithm=="Adadelta"||algorithm=="adadelta"||algorithm=="AdaDelta"||algorithm=="AdaD"||algorithm=="adad"||algorithm=="ADAD")
	{
		fullname="AdaDelta optimizer";
		Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new AdadeltaTrainer(pc,params[0]);
		else
			trainer = new AdadeltaTrainer(pc,params[0],params[1]);
		return trainer;
	}
	if(algorithm=="RMSProp"||algorithm=="rmsprop"||algorithm=="rmsp"||algorithm=="RMSP")
	{
		fullname="RMSProp optimizer";
		Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new RMSPropTrainer(pc,params[0]);
		else if(params.size()==2)
			trainer = new RMSPropTrainer(pc,params[0],params[1]);
		else
			trainer = new RMSPropTrainer(pc,params[0],params[1],params[2]);
		return trainer;
	}
	if(algorithm=="Adam"||algorithm=="adam"||algorithm=="ADAM")
	{
		fullname="Adam optimizer";
		Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new AdamTrainer(pc,params[0]);
		else if(params.size()==2)
			trainer = new AdamTrainer(pc,params[0],params[1]);
		else if(params.size()==3)
			trainer = new AdamTrainer(pc,params[0],params[1],params[2]);
		else
			trainer = new AdamTrainer(pc,params[0],params[1],params[2],params[3]);
		return trainer;
	}
	if(algorithm=="AMSGrad"||algorithm=="Amsgrad"||algorithm=="Amsg"||algorithm=="amsg")
	{
		fullname="AMSGrad optimizer";
		Trainer *trainer=NULL;
		if(params.size()==1)
			trainer = new AmsgradTrainer(pc,params[0]);
		else if(params.size()==2)
			trainer = new AmsgradTrainer(pc,params[0],params[1]);
		else if(params.size()==3)
			trainer = new AmsgradTrainer(pc,params[0],params[1],params[2]);
		else
			trainer = new AmsgradTrainer(pc,params[0],params[1],params[2],params[3]);
		return trainer;
	}
	return NULL;
}


}

#endif

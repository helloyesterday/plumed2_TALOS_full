/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2016-2017 The VES code team
   (see the PEOPLE-VES file at the root of this folder for a list of names)

   See http://www.ves-code.org for more information.

   This file is part of VES code module.

   The VES code module is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   The VES code module is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with the VES code module.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */

#ifdef __PLUMED_HAS_DYNET

#include <random>
#include "Optimizer.h"
#include "CoeffsVector.h"
#include "CoeffsMatrix.h"

#include "core/ActionRegister.h"
#include "core/PlumedMain.h"
#include "core/Atoms.h"
#include "tools/File.h"
#include "tools/DynetTools.h"

namespace PLMD {
namespace ves {

using namespace dytools;

//+PLUMEDOC VES_OPTIMIZER OPT_TALOS
/*
Targeted Adversarial Learning Optimizied Sampling (TALOS)

An optimization algorithm for VES using Wasserstein ganerative adversarial network (W-GAN)

\par Algorithm

\par Examples

\endplumedfile

\endplumedfile

*/
//+ENDPLUMEDOC

class Opt_TALOS :
	public Optimizer,
	public ActionWithArguments
{
private:
	bool is_debug;
	bool opt_const;
	bool read_targetdis;
	bool opt_rc;
	bool do_quadratic;
	bool do_clip;
	bool use_full_loss;
	
	unsigned random_seed;
	
	unsigned narg;
	unsigned target_dim;
	unsigned nbiases;
	unsigned tot_basis;
	unsigned ntarget;
	unsigned nepoch;
	unsigned batch_size;
	unsigned step;
	unsigned counts;
	unsigned update_steps;
	unsigned nw;
	unsigned iter_time;
	unsigned Dw_output;
	unsigned nrcarg;
	unsigned rc_stride;
	unsigned rc_dist_stride;
	
	unsigned update_basis_size;
	unsigned update_target_size;
	unsigned tot_data_size;
	unsigned tot_basis_size;
	unsigned tot_target_size;
	
	const long double PI=3.14159265358979323846264338327950288419716939937510582;
	
	double kB;
	double kBT;
	double beta;
	double sim_temp;
	
	double clip_threshold_bias;
	double clip_threshold_Dw;
	
	float lambda;
	
	float clip_left;
	float clip_right;
	
	std::vector<std::string> arg_names;
	
	std::vector<unsigned> basises_nums;
	std::vector<unsigned> const_id;

	std::vector<bool> args_periodic;

	std::vector<double> grid_min;
	std::vector<double> grid_max;
	std::vector<unsigned> grid_bins;
	std::vector<float> grid_space;

	std::vector<float> input_rc;
	std::vector<float> input_bias;
	std::vector<float> p_rc_sample1;
	std::vector<float> p_rc_sample2;
	std::vector<float> rc_arg_sample;
	
	std::vector<float> ll1;
	std::vector<float> ul1;
	std::vector<float> ll2;
	std::vector<float> ul2;
	std::vector<float> center1;
	std::vector<float> center2;
	std::vector<float> tsp;
	std::vector<float> state_boundary;
	std::vector<float> ps_boundary;
	
	std::vector<std::vector<double>> coe_now;
	
	std::vector<dynet::real> input_target;
	std::vector<dynet::real> target_dis;
	std::vector<dynet::real> basis_values;
	
	std::string algorithm_bias;
	std::string algorithm_Dw;
	
	std::string targetdis_file;
	std::string Dw_file;
	std::string rc_file;
	std::string rc_dist_file;
	std::string debug_file;
	
	IFile itarget;
	OFile orc;
	OFile orcdist;
	OFile odebug;
	
	std::vector<float> lr_bias;
	std::vector<float> lr_Dw;
	std::vector<float> hyper_params_bias;
	std::vector<float> hyper_params_Dw;
	
	ParameterCollection pc_bias;
	ParameterCollection pc_Dw;
	
	Trainer *train_bias;
	Trainer *train_Dw;
	
	Parameter parm_bias;
	Parameter parm_rc_dw;
	Parameter parm_rc_vb;
	MLP nn_Dw;
	
	Value* valueDwLoss;
	Value* valueVbLoss;
	
	float update_Dw(const std::vector<float>&,std::vector<std::vector<float>>&);
	float update_bias(const std::vector<float>&,const std::vector<std::vector<float>>&);
	float update_bias_and_rc(const std::vector<float>&,const std::vector<float>&,
		const std::vector<float>&,const std::vector<float>&,const std::vector<std::vector<float>>&);

public:
	static void registerKeywords(Keywords&);
	explicit Opt_TALOS(const ActionOptions&);
	~Opt_TALOS();
	void update();
	void coeffsUpdate(const unsigned int c_id = 0){}
	void update_rc_target();
	template<class T>
	bool parseVectorAuto(const std::string&, std::vector<T>&,unsigned);
	template<class T>
	bool parseVectorAuto(const std::string&, std::vector<T>&,unsigned,const T&);
};

PLUMED_REGISTER_ACTION(Opt_TALOS,"OPT_TALOS")

void Opt_TALOS::registerKeywords(Keywords& keys) {
  Optimizer::registerKeywords(keys);
  Optimizer::useFixedStepSizeKeywords(keys);
  Optimizer::useMultipleWalkersKeywords(keys);
  Optimizer::useHessianKeywords(keys);
  Optimizer::useMaskKeywords(keys);
  Optimizer::useRestartKeywords(keys);
  Optimizer::useDynamicTargetDistributionKeywords(keys);
  ActionWithArguments::registerKeywords(keys);
  keys.addOutputComponent("DwLoss","default","loss function of the discriminator");
  keys.addOutputComponent("VbLoss","default","loss function of the bias function");
  keys.use("ARG");
  //~ keys.remove("ARG");
  //~ keys.add("compulsory","ARG","the arguments used to set the target distribution");
  keys.remove("STRIDE");
  keys.remove("STEPSIZE");
  keys.add("compulsory","ALGORITHM_BIAS","ADAM","the algorithm to train the neural network of bias function");
  keys.add("compulsory","ALGORITHM_DISCRIM","ADAM","the algorithm to train the discriminator");
  keys.add("compulsory","UPDATE_STEPS","250","the number of steps to update the neural network");
  keys.add("compulsory","EPOCH_NUM","1","number of epoch for each update per walker");
  keys.add("compulsory","HIDDEN_NUMBER","3","the number of hidden layers for discriminator");
  keys.add("compulsory","HIDDEN_LAYER","8","the dimensions of each hidden layer  for discriminator");
  keys.add("compulsory","HIDDEN_ACTIVE","RELU","active function of each hidden layer  for discriminator");
  keys.add("compulsory","DISCRIM_FILE","dw.data","file name of the coefficients of discriminator");
  keys.add("compulsory","DISCRIM_OUTPUT","1","the frequency (how many period of update) to out the coefficients of discriminator");
  keys.add("compulsory","QUADRATIC_FACTOR","0","a factor to adjust the loss function of discriminator");
  keys.add("compulsory","CLIP_RANGE","-0.01,0.01","the range of the value to clip");
  keys.addFlag("FULL_LOSS_FUNCTION",false,"use the full form of loss function to train the bias potential");
  keys.addFlag("OPT_RC",false,"optimize reaction coordinate during the iteration");
  keys.addFlag("NOT_CLIP",false,"do not clip the neural netwrok of discriminator during training");
  //~ keys.add("optional","TARGET_DIM","the dimension of the target order parameters");
  keys.add("optional","RC_INITAL_COEFFS","the initial coefficients of the target order parameters");
  keys.add("optional","TARGETDIST_FILE","read target distribution from file");
  keys.add("optional","OPT_TARGET_FILE","the file to output the distribution optimized reaction coordinate");
  keys.add("optional","OPT_TARGET_STRIDE","the frequency to output the distribution optimized reaction coordinate");
  keys.add("optional","OPT_RC_FILE","the file to output the parameters of optimized reaction coordinate");
  keys.add("optional","OPT_RC_STRIDE","the frequency to output the parameters of optimized reaction coordinate");
  keys.add("optional","STATE1_LL","the lower bounds of state 1");
  keys.add("optional","STATE1_UL","the upper bounds of state 1");
  keys.add("optional","STATE2_LL","the upper bounds of state 2");
  keys.add("optional","STATE2_UL","the upper bounds of state 2");
  keys.add("optional","STATE1_CENTER","the center of state 1 used to build target distribution");
  keys.add("optional","STATE2_CENTER","the center of state 2 used to build target distribution");
  keys.add("optional","TS_CENTER","the center of transiiton state used to build target distribution");
  keys.add("optional","GRID_MIN","the lower bounds used to calculate the target distribution");
  keys.add("optional","GRID_MAX","the upper bounds used to calculate the target distribution");
  keys.add("optional","GRID_BINS","the number of bins used to set the target distribution");
  keys.add("optional","ARG_PERIODIC","if the arguments are periodic or not");
  keys.addFlag("OPTIMIZE_CONSTANT_PARAMETER",false,"also to optimize the constant part of the basis functions");
  keys.add("optional","LEARN_RATE_BIAS","the learning rate for training the neural network of bias function");
  keys.add("optional","LEARN_RATE_DISCRIM","the learning rate for training the discriminator");
  keys.add("optional","HYPER_PARAMS_BIAS","other hyperparameters for training the neural network of bias function");
  keys.add("optional","HYPER_PARAMS_DISCRIM","other hyperparameters for training the discriminator");
  keys.add("optional","CLIP_THRESHOLD_BIAS","the clip threshold for training the neural network of bias function");
  keys.add("optional","CLIP_THRESHOLD_DISCRIM","the clip threshold for training the discriminator");
  keys.add("optional","SIM_TEMP","the simulation temperature");
  keys.add("optional","DEBUG_FILE","the file to debug");
}


Opt_TALOS::~Opt_TALOS() {
	delete train_Dw;
	delete train_bias;
	if(opt_rc)
	{
		if(rc_file.size()>0)
			orc.close();
		if(rc_dist_file.size()>0)
			orcdist.close();
	}
	if(is_debug)
		odebug.close();
}


Opt_TALOS::Opt_TALOS(const ActionOptions&ao):
  PLUMED_VES_OPTIMIZER_INIT(ao),
  ActionWithArguments(ao),
  is_debug(false),
  read_targetdis(false),
  random_seed(0),
  narg(getNumberOfArguments()),
  step(0),
  counts(0),
  nw(1),
  iter_time(0),
  rc_stride(1),
  rc_dist_stride(1),
  args_periodic(getNumberOfArguments(),false),
  grid_space(getNumberOfArguments()),
  train_bias(NULL),
  train_Dw(NULL)
{
	random_seed=0;
	if(useMultipleWalkers())
	{
		if(comm.Get_rank()==0)
		{
			if(multi_sim_comm.Get_rank()==0)
			{
				std::random_device rd;
				random_seed=rd();
			}
			multi_sim_comm.Barrier();
			multi_sim_comm.Bcast(random_seed,0);
		}
		comm.Barrier();
		comm.Bcast(random_seed,0);
	}
	else
	{
		if(comm.Get_rank()==0)
		{
			std::random_device rd;
			random_seed=rd();
		}
		comm.Barrier();
		comm.Bcast(random_seed,0);
	}
	
	int cc=1;
	char pp[]="plumed";
	char *vv[]={pp};
	char** ivv=vv;
	DynetParams params = extract_dynet_params(cc,ivv,true);

	params.random_seed=random_seed;
	
	dynet::initialize(params);
	
	for(unsigned i=0;i!=narg;++i)
	{
		arg_names.push_back(getPntrToArgument(i)->getName());
		//~ log.printf("  %d with argument names: %s\n",int(i),arg_names[i].c_str());
	}
	
	setStride(1);
	parse("UPDATE_STEPS",update_steps);
	
	parse("ALGORITHM_BIAS",algorithm_bias);
	parse("ALGORITHM_DISCRIM",algorithm_Dw);
	
	parseVector("LEARN_RATE_BIAS",lr_bias);
	parseVector("LEARN_RATE_DISCRIM",lr_Dw);
	
	std::vector<float> other_params_bias,other_params_Dw;
	parseVector("HYPER_PARAMS_BIAS",other_params_bias);
	parseVector("HYPER_PARAMS_DISCRIM",other_params_Dw);
	
	unsigned nhidden=0;
	parse("HIDDEN_NUMBER",nhidden);
	
	std::vector<unsigned> hidden_layers;
	std::vector<std::string> hidden_active;
	parseVectorAuto("HIDDEN_LAYER",hidden_layers,nhidden);
	parseVectorAuto("HIDDEN_ACTIVE",hidden_active,nhidden);
	
	do_clip=true;
	bool no_clip;
	parseFlag("NOT_CLIP",no_clip);
	if(no_clip)
		do_clip=false;
	else
	{
		std::vector<float> clips; 
		parseVector("CLIP_RANGE",clips);
		plumed_massert(clips.size()>=2,"CLIP_RANGE should has left and right values");
		clip_left=clips[0];
		clip_right=clips[1];
		plumed_massert(clip_right>clip_left,"Clip left value should less than clip right value");
	}
	
	parse("DISCRIM_FILE",Dw_file);
	parse("DISCRIM_OUTPUT",Dw_output);
	
	do_quadratic=false;
	parse("QUADRATIC_FACTOR",lambda);
	if(lambda>1e-38)
		do_quadratic=true;
		
	parseFlag("FULL_LOSS_FUNCTION",use_full_loss);
	
	parseFlag("OPTIMIZE_CONSTANT_PARAMETER",opt_const);
	
	nbiases=numberOfCoeffsSets();
	tot_basis=0;
	for(unsigned i=0;i!=nbiases;++i)
	{
		unsigned nset=Coeffs(i).getSize();
		basises_nums.push_back(nset);
		if(!opt_const)
			--nset;
		tot_basis+=nset;
	}
	
	std::vector<float> init_coe;
	std::string fullname_bias,fullname_Dw;
	
	if(lr_bias.size()==0)
	{
		train_bias=new_traniner(algorithm_bias,pc_bias,fullname_bias);
	}
	else
	{
		if(algorithm_bias=="CyclicalSGD"||algorithm_bias=="cyclicalSGD"||algorithm_bias=="cyclicalsgd"||algorithm_bias=="CSGD"||algorithm_bias=="csgd")
		{
			plumed_massert(lr_bias.size()==2,"The CyclicalSGD algorithm need two learning rates");
		}
		else
		{
			plumed_massert(lr_bias.size()==1,"The "+algorithm_bias+" algorithm need only one learning rates");
		}
		
		hyper_params_bias.insert(hyper_params_bias.end(),lr_bias.begin(),lr_bias.end());
		
		if(other_params_bias.size()>0)
			hyper_params_bias.insert(hyper_params_bias.end(),other_params_bias.begin(),other_params_bias.end());
		
		train_bias=new_traniner(algorithm_bias,pc_bias,hyper_params_bias,fullname_bias);
	}

	if(lr_Dw.size()==0)
	{
		train_Dw=new_traniner(algorithm_Dw,pc_Dw,fullname_Dw);
	}
	else
	{
		if(algorithm_Dw=="CyclicalSGD"||algorithm_Dw=="cyclicalSGD"||algorithm_Dw=="cyclicalsgd"||algorithm_Dw=="CSGD"||algorithm_Dw=="csgd")
		{
			plumed_massert(lr_Dw.size()==2,"The CyclicalSGD algorithm need two learning rates");
		}
		else
		{
			plumed_massert(lr_Dw.size()==1,"The "+algorithm_Dw+" algorithm need only one learning rates");
		}
		
		hyper_params_Dw.insert(hyper_params_Dw.end(),lr_Dw.begin(),lr_Dw.end());
		
		if(other_params_Dw.size()>0)
			hyper_params_Dw.insert(hyper_params_Dw.end(),other_params_Dw.begin(),other_params_Dw.end());
		
		train_Dw=new_traniner(algorithm_Dw,pc_Dw,hyper_params_Dw,fullname_Dw);
	}
	
	clip_threshold_bias=train_bias->clip_threshold;
	clip_threshold_Dw=train_Dw->clip_threshold;
	
	parse("CLIP_THRESHOLD_BIAS",clip_threshold_bias);
	parse("CLIP_THRESHOLD_DISCRIM",clip_threshold_Dw);
	
	train_bias->clip_threshold = clip_threshold_bias;
	train_Dw->clip_threshold = clip_threshold_Dw;
	
	parseFlag("OPT_RC",opt_rc);
	
	if(opt_rc)
		target_dim=1;
	else
		target_dim=narg;

	unsigned ldim=target_dim;
	for(unsigned i=0;i!=nhidden;++i)
	{
		nn_Dw.append(pc_Dw,Layer(ldim,hidden_layers[i],activation_function(hidden_active[i]),0));
		ldim=hidden_layers[i];
	}
	nn_Dw.append(pc_Dw,Layer(ldim,1,LINEAR,0));
	
	parm_bias=pc_bias.add_parameters({1,tot_basis});
	
	if(opt_rc)
	{
		parm_rc_dw=pc_bias.add_parameters({target_dim,narg});
		parm_rc_vb=pc_bias.add_parameters({target_dim});
		
		parse("OPT_TARGET_FILE",rc_dist_file);
		parse("OPT_TARGET_STRIDE",rc_dist_stride);
		
		if(rc_dist_file.size()>0)
		{
			orcdist.link(*this);
			orcdist.open(rc_dist_file.c_str());
		}
		
		rc_file="rc.data";
		parse("OPT_RC_FILE",rc_file);
		parse("OPT_RC_STRIDE",rc_stride);
		
		if(rc_file.size()>0)
		{
			orc.link(*this);
			orc.open(rc_file.c_str());
		}
		
		plumed_massert(narg>1,"if you want to optimize the reaction coordinate, the number of arguments must larger than 1");
		parseVector("STATE1_LL",ll1);
		plumed_massert(ll1.size()==narg,"the number of STATE1_LL must equal to the number of arguments");
		state_boundary.insert(state_boundary.end(),ll1.begin(),ll1.end());
		
		parseVector("STATE1_UL",ul1);
		plumed_massert(ul1.size()==narg,"the number of STATE1_UL must equal to the number of arguments");
		state_boundary.insert(state_boundary.end(),ul1.begin(),ul1.end());
		
		parseVector("STATE2_LL",ll2);
		plumed_massert(ll2.size()==narg,"the number of STATE2_LL must equal to the number of arguments");
		state_boundary.insert(state_boundary.end(),ll2.begin(),ll2.end());
		
		parseVector("STATE2_UL",ul2);
		plumed_massert(ul2.size()==narg,"the number of STATE2_UL must equal to the number of arguments");
		state_boundary.insert(state_boundary.end(),ul2.begin(),ul2.end());
		
		parseVector("STATE1_CENTER",center1);
		plumed_massert(center1.size()==narg,"the number of STATE1_CENTER must equal to the number of arguments");
		ps_boundary.insert(ps_boundary.end(),center1.begin(),center1.end());
		
		parseVector("STATE2_CENTER",center2);
		plumed_massert(center2.size()==narg,"the number of STATE2_CENTER must equal to the number of arguments");
		ps_boundary.insert(ps_boundary.end(),center2.begin(),center2.end());
		
		parseVector("TS_CENTER",tsp);
		plumed_massert(tsp.size()==narg,"the number of TS_CENTER must equal to the number of arguments");
		ps_boundary.insert(ps_boundary.end(),tsp.begin(),tsp.end());
	}
	
	init_coe.resize(tot_basis);
	std::vector<float> init_rc_dw(narg);
	std::vector<float> init_rc_vb(1);
	
	std::vector<float> params_Dw(nn_Dw.parameters_number());
	
	if(useMultipleWalkers())
	{
		if(comm.Get_rank()==0)
		{
			if(multi_sim_comm.Get_rank()==0)
			{
				init_coe=as_vector(*parm_bias.values());
				params_Dw=nn_Dw.get_parameters();
				if(opt_rc)
				{
					init_rc_dw=as_vector(*parm_rc_dw.values());
					init_rc_vb=as_vector(*parm_rc_vb.values());
				}
			}
			multi_sim_comm.Barrier();
			multi_sim_comm.Bcast(init_coe,0);
			multi_sim_comm.Bcast(params_Dw,0);
			if(opt_rc)
			{
				multi_sim_comm.Bcast(init_rc_dw,0);
				multi_sim_comm.Bcast(init_rc_vb,0);
			}
		}
		comm.Bcast(init_coe,0);
		comm.Bcast(params_Dw,0);
		if(opt_rc)
		{
			comm.Bcast(init_rc_dw,0);
			comm.Bcast(init_rc_vb,0);
		}
	}
	else
	{
		if(comm.Get_rank()==0)
		{
			init_coe=as_vector(*parm_bias.values());
			params_Dw=nn_Dw.get_parameters();
			if(opt_rc)
			{
				init_rc_dw=as_vector(*parm_rc_dw.values());
				init_rc_vb=as_vector(*parm_rc_vb.values());
			}
		}
		comm.Barrier();
		comm.Bcast(init_coe,0);
		comm.Bcast(params_Dw,0);
		if(opt_rc)
		{
			comm.Bcast(init_rc_dw,0);
			comm.Bcast(init_rc_vb,0);
		}
	}
	
	parseVector("RC_INITAL_COEFFS",init_rc_dw);
	
	parm_bias.set_value(init_coe);
	nn_Dw.set_parameters(params_Dw);
	if(opt_rc)
	{
		std::vector<float> input_coe;
		parseVector("RC_INITAL_COEFFS",input_coe);
		
		if(input_coe.size()>0)
		{
			if(input_coe.size()==narg)
				init_rc_dw=input_coe;
			else if(input_coe.size()==narg+1)
			{
				init_rc_vb={input_coe.back()};
				input_coe.pop_back();
				init_rc_dw=input_coe;
			}
			else
				plumed_merror("the number of RC_INITAL_COEFFS must be equal to or larger than the arguments number");
		}
		
		parm_rc_dw.set_value(init_rc_dw);
		parm_rc_vb.set_value(init_rc_vb);
		
		orc.printField("time",getTime());
		for(unsigned i=0;i!=init_rc_dw.size();++i)
		{
			std::string lab="W"+std::to_string(i);
			orc.printField(lab,init_rc_dw[i]);
		}
		orc.printField("b",init_rc_vb[0]);
		orc.printField();
		orc.flush();
	}

	coe_now.resize(0);
	unsigned id=0;
	for(unsigned i=0;i!=nbiases;++i)
	{
		std::vector<double> coe(basises_nums[i]);
		for(unsigned j=0;j!=basises_nums[i];++j)
		{
			if(j==0&&(!opt_const))
				coe[j]=0;
			else
				coe[j]=init_coe[id++];
		}
		coe_now.push_back(coe);
		Coeffs(i).setValues(coe);
	}
	
	kB=plumed.getAtoms().getKBoltzmann();
	
	sim_temp=-1;
	parse("SIM_TEMP",sim_temp);
	if(sim_temp>0)
		kBT=kB*sim_temp;
	else
	{
		kBT=plumed.getAtoms().getKbT();
		sim_temp=kBT/kB;
	}
	beta=1.0/kBT;
	
	parse("DEBUG_FILE",debug_file);
	if(debug_file.size()>0)
	{
		is_debug=true;
		odebug.link(*this);
		odebug.open(debug_file);
		odebug.fmtField(" %f");
		//~ odebug.addConstantField("ITERATION");
	}
	
	parse("EPOCH_NUM",nepoch);
	plumed_massert(nepoch>0,"EPOCH_NUM must be larger than 0!");
	batch_size=update_steps/nepoch;
	plumed_massert((update_steps-batch_size*nepoch)==0,"UPDATE_STEPS must be divided exactly by EPOCH_NUM");
	
	parse("TARGETDIST_FILE",targetdis_file);
	if(targetdis_file.size()>0)
		read_targetdis=true;
	
	std::vector<std::string> arg_perd_str;
	ntarget=1;
	if(read_targetdis)
	{
		itarget.link(*this);
		itarget.open(targetdis_file.c_str());
		itarget.allowIgnoredFields();
		
		for(unsigned i=0;i!=narg;++i)
		{
			if(!itarget.FieldExist(arg_names[i]))
				plumed_merror("Cannot found Field \""+arg_names[i]+"\"");
			if(!itarget.FieldExist("targetdist"))
				plumed_merror("Cannot found Field \"targetdist\"");

			double fvv=0;
			if(itarget.FieldExist("min_"+arg_names[i]))
				itarget.scanField("min_"+arg_names[i],fvv);
			grid_min.push_back(fvv);
			
			if(itarget.FieldExist("max_"+arg_names[i]))
				itarget.scanField("max_"+arg_names[i],fvv);
			grid_max.push_back(fvv);
			
			int bin=0;
			if(itarget.FieldExist("nbins_"+arg_names[i]))
				itarget.scanField("nbins_"+arg_names[i],bin);
			else
				plumed_merror("Cannot found Field \"nbins_"+arg_names[i]+"\"");
			grid_bins.push_back(bin);
			ntarget*=bin;
			
			std::string ip="false";
			if(itarget.FieldExist("periodic_"+arg_names[i]))
				itarget.scanField("periodic_"+arg_names[i],ip);
			arg_perd_str.push_back(ip);
		}
	}
	else
	{
		parseVectorAuto("GRID_MIN",grid_min,target_dim);
		parseVectorAuto("GRID_MAX",grid_max,target_dim);
		parseVectorAuto("GRID_BINS",grid_bins,target_dim,unsigned(500));
		parseVectorAuto("ARG_PERIODIC",arg_perd_str,target_dim,std::string("false"));
		
		for(unsigned i=0;i!=target_dim;++i)
		{
			if(arg_perd_str[i]=="True"||arg_perd_str[i]=="true"||arg_perd_str[i]=="TRUE")
				args_periodic[i]=true;
			else if(arg_perd_str[i]=="False"||arg_perd_str[i]=="false"||arg_perd_str[i]=="FALSE")
				args_periodic[i]=false;
			else
				plumed_merror("Cannot understand the ARG_PERIODIC type: "+arg_perd_str[i]);
				
			grid_space[i]=(grid_max[i]-grid_min[i])/grid_bins[i];
			if(!args_periodic[i])
				++grid_bins[i];
			ntarget*=grid_bins[i];
		}
	}
	comm.Barrier();
	
	std::vector<std::vector<float>> target_points;
	if(read_targetdis)
	{
		double sum=0;
		for(unsigned i=0;i!=ntarget;++i)
		{
			std::vector<float> vtt;
			for(unsigned j=0;j!=narg;++j)
			{
				double tt;
				itarget.scanField(arg_names[j],tt);
				vtt.push_back(tt);
				input_target.push_back(tt);
			}
			double p0;
			itarget.scanField("targetdist",p0);
			itarget.scanField();
			
			sum+=p0;
			target_dis.push_back(p0);
			target_points.push_back(vtt);
		}
		itarget.close();
		comm.Barrier();
		for(unsigned i=0;i!=ntarget;++i)
			target_dis[i]/=(sum/ntarget);
	}
	else if(opt_rc)
	{
		target_dis.resize(ntarget);
		update_rc_target();
		for(unsigned i=0;i!=ntarget;++i)
		{
			target_points.push_back({input_target[i]});
			orcdist.printField("RC",input_target[i]);
			orcdist.printField("targetdist",target_dis[i]);
			orcdist.printField();
		}
		orcdist.printField();
	}
	else
	{
		target_dis.assign(ntarget,1.0);
		std::vector<unsigned> argid(narg,0);
		for(unsigned i=0;i!=ntarget;++i)
		{
			std::vector<float> vtt;
			for(unsigned j=0;j!=narg;++j)
			{
				float tt=grid_min[j]+grid_space[j]*argid[j];
				if(args_periodic[j])
					tt+=grid_space[j]/2.0;
				vtt.push_back(tt);
				input_target.push_back(tt);
				if(j==0)
					++argid[j];
				else if(argid[j-1]==grid_bins[j-1])
				{
					++argid[j];
					argid[j-1]%=grid_bins[j-1];
				}
			}
			target_points.push_back(vtt);
		}
	}
	
	update_basis_size=update_steps*tot_basis;
	update_target_size=update_steps*target_dim;
	
	tot_data_size=update_steps;
	tot_basis_size=update_basis_size;
	tot_target_size=update_target_size;
	
	if(useMultipleWalkers())
	{
		if(comm.Get_rank()==0)
			nw=multi_sim_comm.Get_size();
		comm.Bcast(nw,0);
		
		nepoch*=nw;	
		tot_data_size*=nw;
		tot_basis_size*=nw;
		tot_target_size*=nw;
	}
	
	addComponent("DwLoss"); componentIsNotPeriodic("DwLoss");
	valueDwLoss=getPntrToComponent("DwLoss");
	addComponent("VbLoss"); componentIsNotPeriodic("VbLoss");
	valueVbLoss=getPntrToComponent("VbLoss");
	
	turnOffHessian();
	checkRead();
	
	log.printf("  Targeted adversarial learning otimizied sampling (TALOS)\n");
	log.printf("  with random seed: %s\n",std::to_string(random_seed).c_str());
	log.printf("  with lower boundary of the grid:");
	for(unsigned i=0;i!=grid_min.size();++i)
		log.printf(" %f",grid_min[i]);
	log.printf("\n");
	log.printf("  with upper boundary of the grid:");
	for(unsigned i=0;i!=grid_max.size();++i)
		log.printf(" %f",grid_max[i]);
	log.printf("\n");
	log.printf("  with grid bins:");
	for(unsigned i=0;i!=grid_bins.size();++i)
		log.printf(" %d",grid_bins[i]);
	log.printf("\n");
	if(read_targetdis)
	{
		log.printf("  with target distribution read from file: %s\n",targetdis_file.c_str());
		for(unsigned i=0;i!=ntarget;++i)
		{
			log.printf("    target %d:",int(i));
			for(unsigned j=0;j!=narg;++j)
				log.printf(" %f,",target_points[i][j]);
			log.printf(" with %f\n",target_dis[i]);
		}
	}
	else
	{
		log.printf("  with uniform target distribution\n");
		for(unsigned i=0;i!=narg;++i)
		log.printf("    %d Argument %s: from %f to %f with %d bins\n",int(i),arg_names[i].c_str(),grid_min[i],grid_max[i],int(grid_bins[i]));
	}
	log.printf("  with total target segments: %d\n",int(ntarget));
	
	log.printf("  with simulation temperature: %f\n",sim_temp);
	log.printf("  with boltzmann constant: %f\n",kB);
	log.printf("  with beta (1/kT): %f\n",beta);
	log.printf("  with bias functions: %d\n",int(nbiases));
	log.printf("  with total optimized coefficients: %d\n",int(tot_basis));
	if(is_debug)
	{
		id=0;
		for(unsigned i=0;i!=nbiases;++i)
		{
			float coe;
			for(unsigned j=0;j!=basises_nums[i];++j)
			{		
				if(j==0&&(!opt_const))
					coe=0;
				else
					coe=init_coe[id++];
				log.printf("    Bias %d with %d initial coefficient: %e and %e\n",int(i),int(j),coe,Coeffs(i).getValue(j));
			}
		}
	}
	
	log.printf("  Use %s to train the discriminator\n",fullname_Dw.c_str());
	if(do_quadratic)
		log.printf("    with quadratic term factor: %f\n",lambda);
	else
		log.printf("    without using quadratic term factor.\n");
	if(lr_Dw.size()>0)
	{
		log.printf("    with learning rates:");
		for(unsigned i=0;i!=lr_Dw.size();++i)
			log.printf(" %f",lr_Dw[i]);
		log.printf("\n");
		if(other_params_Dw.size()>0)
		{
			log.printf("    with other hyperparameters:");
			for(unsigned i=0;i!=other_params_Dw.size();++i)
				log.printf(" %f",other_params_Dw[i]);
			log.printf("\n");
		}
	}
	else
		log.printf("    with default hyperparameters\n");
	log.printf("    with clip threshold: %f\n",clip_threshold_Dw);
	log.printf("    with hidden layers: %d\n",int(nhidden));
	for(unsigned i=0;i!=nhidden;++i)
		log.printf("      Hidden layer %d with dimension %d and activation function \"%s\"\n",int(i),int(hidden_layers[i]),hidden_active[i].c_str());
	if(do_clip)
		log.printf("    with clip range %f to %f\n",clip_left,clip_right);
	else
		log.printf("    without cliping\n");
	
	log.printf("  Use %s to train the neural network of bias function\n",fullname_bias.c_str());
	if(use_full_loss)
		log.printf("  with the full form of the loss function.\n");
	else
		log.printf("  with the approximated form of the loss function.\n");
	if(lr_bias.size()>0)
	{
		log.printf("    with learning rates:");
		for(unsigned i=0;i!=lr_bias.size();++i)
			log.printf(" %f",lr_bias[i]);
		log.printf("\n");
		if(other_params_bias.size()>0)
		{
			log.printf("    with other hyperparameters:");
			for(unsigned i=0;i!=other_params_bias.size();++i)
				log.printf(" %f",other_params_bias[i]);
			log.printf("\n");
		}
	}
	else
		log.printf("    with default hyperparameters\n");
	log.printf("    with clip threshold: %f\n",clip_threshold_bias);
	
	log.printf("  Use epoch number %d and batch size %d to update the two networks\n",int(nepoch),int(batch_size));
	log << plumed.cite("Zhang, Yang and Noe");
	log.printf("\n");
}

void Opt_TALOS::update()
{
	bool irc1=true;
	bool irc2=true;
	std::vector<float> varg;
	if(is_debug)
		odebug.printField("time",getTime());
	for(unsigned i=0;i!=narg;++i)
	{
		float val=getArgument(i);
		
		if(opt_rc)
		{
			varg.push_back(val);
			rc_arg_sample.push_back(val);
		}
		else
			input_rc.push_back(val);
		
		if(opt_rc)
		{
			if(val<ll1[i]||val>ul1[i])
				irc1=false;
			if(val<ll2[i]||val>ul2[i])
				irc2=false;
		}
		
		if(is_debug)
		{
			std::string ff="ARG"+std::to_string(i);
			odebug.printField(ff,val);
			//~ if(opt_rc)
			//~ {
				//~ ff="LL1_"+std::to_string(i);
				//~ odebug.printField(ff,ll1[i]);
				//~ ff="UL1_"+std::to_string(i);
				//~ odebug.printField(ff,ul1[i]);
				//~ ff="LL2_"+std::to_string(i);
				//~ odebug.printField(ff,ll2[i]);
				//~ ff="UL2_"+std::to_string(i);
				//~ odebug.printField(ff,ul2[i]);
			//~ }
		}
	}
	if(opt_rc)
	{
		ComputationGraph cg;
		Expression W = parameter(cg,parm_rc_dw);
		Expression b = parameter(cg,parm_rc_vb);
		Expression x = input(cg,{narg},&varg);
		Expression y_pred = tanh( W * x + b);
		
		std::vector<float> vrc=as_vector(cg.forward(y_pred));
		
		input_rc.push_back(vrc[0]);
		
		if(irc1)
			p_rc_sample1.push_back(1);
		else
			p_rc_sample1.push_back(0);
			
		if(irc2)
			p_rc_sample2.push_back(1);
		else
			p_rc_sample2.push_back(0);
			
		if(is_debug)
		{
			odebug.printField("RC",vrc[0]);
			odebug.printField("ST1",p_rc_sample1.back());
			odebug.printField("ST2",p_rc_sample2.back());
		}
	}
	
	std::vector<float> debug_vec;
	for(unsigned i=0;i!=nbiases;++i)
	{
		std::vector<double> bsvalues;
		getBiasPntrs()[i]->getBasisSetValues(bsvalues);
		
		unsigned begid=1;
		if(opt_const)
			begid=0;
		
		for(unsigned j=begid;j!=bsvalues.size();++j)
		{
			input_bias.push_back(bsvalues[j]);
			//~ if(is_debug)
				//~ debug_vec.push_back(bsvalues[j]);
		}
	}
	
	if(is_debug)
	{
		//~ std::vector<float> pred(tot_basis,0);
		//~ std::vector<float> ww(tot_basis,0);
		//~ std::vector<float> xx(tot_basis,0);
		//~ if(comm.Get_rank()==0)
		//~ {
			//~ ComputationGraph cg;
			//~ Expression W = parameter(cg,parm_bias);
			//~ Expression x = input(cg,{tot_basis},&debug_vec);
			//~ Expression y_pred = W * x;
			
			//~ if(counts==0)
			//~ {
				//~ ww=as_vector(W.value());
				//~ xx=as_vector(x.value());
			//~ }
			//~ pred=as_vector(cg.forward(y_pred));
		//~ }
		//~ comm.Barrier();
		//~ comm.Bcast(pred,0);
		//~ if(counts==0)
		//~ {
			//~ comm.Bcast(ww,0);
			//~ comm.Bcast(xx,0);
		//~ }
		
		//~ for(unsigned i=0;i!=pred.size();++i)
		//~ {
			//~ std::string ff="PRED"+std::to_string(i);
			//~ odebug.printField(ff,pred[i]);
		//~ }
		//~ 
		odebug.printField();
		odebug.flush();
		
		//~ if(counts==0)
		//~ {
			//~ odebug.printField("time",getTime());
			//~ for(unsigned i=0;i!=ww.size();++i)
			//~ {
				//~ std::string ff="W"+std::to_string(i);
				//~ odebug.printField(ff,ww[i]);
			//~ }
			//~ odebug.printField();
			//~ 
			//~ odebug.printField("time",getTime());
			//~ for(unsigned i=0;i!=xx.size();++i)
			//~ {
				//~ std::string ff="x"+std::to_string(i);
				//~ odebug.printField(ff,xx[i]);
			//~ }
			//~ odebug.printField();
			//~ odebug.flush();
		//~ }
	}

	++counts;
	++step;
	
	if(step%update_steps==0&&counts>0)
	{
		if(input_rc.size()!=update_target_size)
			plumed_merror("ERROR! The size of the input_rc mismatch: "+std::to_string(input_rc.size()));
		if(input_bias.size()!=update_basis_size)
			plumed_merror("ERROR! The size of the input_bias mismatch: "+std::to_string(input_bias.size()));

		double dwloss=0;
		double vbloss=0;
		std::vector<float> new_coe(tot_basis,0);
		std::vector<std::vector<float>> vec_fw;
		std::vector<float> params_Dw(nn_Dw.parameters_number(),0);
		
		std::vector<float> new_rc_dw;
		std::vector<float> new_rc_vb;
		if(opt_rc)
		{
			new_rc_dw.resize(target_dim*narg);
			new_rc_vb.resize(target_dim);
		}
		
		if(useMultipleWalkers())
		{
			if(comm.Get_rank()==0)
				multi_sim_comm.Sum(counts);
			comm.Bcast(counts,0);

			std::vector<float> all_input_rc;
			std::vector<float> all_input_bias;
			
			all_input_rc.resize(tot_target_size,0);
			all_input_bias.resize(tot_basis_size,0);
			
			std::vector<float> all_rc_arg_sample;
			std::vector<float> all_p_rc_sample1;
			std::vector<float> all_p_rc_sample2;
			
			if(opt_rc)
			{
				all_rc_arg_sample.resize(tot_data_size*narg,0);
				all_p_rc_sample1.resize(tot_data_size,0);
				all_p_rc_sample2.resize(tot_data_size,0);
			}
			
			if(comm.Get_rank()==0)
			{
				multi_sim_comm.Allgather(input_rc,all_input_rc);
				multi_sim_comm.Allgather(input_bias,all_input_bias);
				if(opt_rc)
				{
					multi_sim_comm.Allgather(rc_arg_sample,all_rc_arg_sample);
					multi_sim_comm.Allgather(p_rc_sample1,all_p_rc_sample1);
					multi_sim_comm.Allgather(p_rc_sample2,all_p_rc_sample2);
				}
			}
			comm.Bcast(all_input_rc,0);
			comm.Bcast(all_input_bias,0);
			if(opt_rc)
			{
				comm.Bcast(all_rc_arg_sample,0);
				comm.Bcast(all_p_rc_sample1,0);
				comm.Bcast(all_p_rc_sample2,0);
			}

			if(comm.Get_rank()==0)
			{
				if(multi_sim_comm.Get_rank()==0)
				{
					dwloss=update_Dw(all_input_rc,vec_fw);
					if(opt_rc)
						vbloss=update_bias_and_rc(all_input_bias,
							all_rc_arg_sample,all_p_rc_sample1,
							all_p_rc_sample2,vec_fw);
					else
						vbloss=update_bias(all_input_bias,vec_fw);
					vec_fw.resize(0);
					new_coe=as_vector(*parm_bias.values());
					params_Dw=nn_Dw.get_parameters();
					if(opt_rc)
					{
						new_rc_dw=as_vector(*parm_rc_dw.values());
						new_rc_vb=as_vector(*parm_rc_vb.values());
					}
				}
				multi_sim_comm.Barrier();
				multi_sim_comm.Bcast(dwloss,0);
				multi_sim_comm.Bcast(vbloss,0);
				multi_sim_comm.Bcast(new_coe,0);
				multi_sim_comm.Bcast(params_Dw,0);
				if(opt_rc)
				{
					multi_sim_comm.Bcast(new_rc_dw,0);
					multi_sim_comm.Bcast(new_rc_vb,0);
				}
			}
			comm.Barrier();
			comm.Bcast(dwloss,0);
			comm.Bcast(vbloss,0);
			comm.Bcast(new_coe,0);
			comm.Bcast(params_Dw,0);
			if(opt_rc)
			{
				comm.Bcast(new_rc_dw,0);
				comm.Bcast(new_rc_vb,0);
			}
		}
		else
		{
			if(comm.Get_rank()==0)
			{
				dwloss=update_Dw(input_rc,vec_fw);
				if(opt_rc)
					vbloss=update_bias_and_rc(input_bias,rc_arg_sample,
						p_rc_sample1,p_rc_sample2,vec_fw);
				else
					vbloss=update_bias(input_bias,vec_fw);
				vec_fw.resize(0);
				new_coe=as_vector(*parm_bias.values());
				params_Dw=nn_Dw.get_parameters();
				if(opt_rc)
				{
					new_rc_dw=as_vector(*parm_rc_dw.values());
					new_rc_vb=as_vector(*parm_rc_vb.values());
				}
			}
			comm.Barrier();
			comm.Bcast(dwloss,0);
			comm.Bcast(vbloss,0);
			comm.Bcast(new_coe,0);
			comm.Bcast(params_Dw,0);
			if(opt_rc)
			{
				comm.Bcast(new_rc_dw,0);
				comm.Bcast(new_rc_vb,0);
			}
		}
		
		parm_bias.set_value(new_coe);
		nn_Dw.set_parameters(params_Dw);
		if(opt_rc)
		{
			parm_rc_dw.set_value(new_rc_dw);
			parm_rc_vb.set_value(new_rc_vb);
			if(iter_time%rc_stride==0)
			{
				orc.printField("time",getTime());
				for(unsigned i=0;i!=new_rc_dw.size();++i)
				{
					std::string lab="W"+std::to_string(i);
					orc.printField(lab,new_rc_dw[i]);
				}
				orc.printField("b",new_rc_vb[0]);
				orc.printField();
				orc.flush();
			}
			update_rc_target();
			if(iter_time%rc_dist_stride==0)
			{
				for(unsigned i=0;i!=ntarget;++i)
				{
					orcdist.printField("RC",input_target[i]);
					orcdist.printField("targetdist",target_dis[i]);
					orcdist.printField();
				}
				orcdist.printField();
			}
		}

		valueDwLoss->set(dwloss);
		valueVbLoss->set(vbloss);
		
		input_rc.resize(0);
		input_bias.resize(0);
		counts=0;
		if(opt_rc)
		{
			rc_arg_sample.resize(0);
			p_rc_sample1.resize(0);
			p_rc_sample2.resize(0);
		}
		
		if(comm.Get_rank()==0&&multi_sim_comm.Get_rank()==0&&iter_time%Dw_output==0)
		{
			TextFileSaver saver(Dw_file);
			saver.save(pc_Dw);
		}
		
		++iter_time;
		
		//~ if(is_debug)
		//~ {
			//~ odebug.printField("ITERATION",int(iter_time));
			//~ odebug.printField("time",getTime());
			//~ for(unsigned i=0;i!=new_coe.size();++i)
			//~ {
				//~ odebug.printField("Coe"+std::to_string(i),new_coe[i]);
			//~ }
			
			//~ odebug.printField();
			//~ odebug.flush();
		//~ }

		coe_now.resize(0);
		unsigned id=0;
		for(unsigned i=0;i!=nbiases;++i)
		{
			std::vector<double> coe(basises_nums[i]);
			for(unsigned j=0;j!=basises_nums[i];++j)
			{
				if(j==0&&(!opt_const))
					coe[j]=0;
				else
					coe[j]=new_coe[id++];
			}
			coe_now.push_back(coe);
			Coeffs(i).setValues(coe);

			unsigned int curr_iter = getIterationCounter()+1;
			double curr_time = getTime();
			getCoeffsPntrs()[i]->setIterationCounterAndTime(curr_iter,curr_time);
		}

		increaseIterationCounter();
		updateOutputComponents();
		for(unsigned int i=0; i<numberOfCoeffsSets(); i++) {
			writeOutputFiles(i);
		}
		if(TartgetDistStride()>0 && getIterationCounter()%TartgetDistStride()==0) {
			for(unsigned int i=0; i<numberOfBiases(); i++) {
				if(DynamicTargetDists()[i]) {
					getBiasPntrs()[i]->updateTargetDistributions();
				}
			}
		}
		if(StrideReweightFactor()>0 && getIterationCounter()%StrideReweightFactor()==0) {
			for(unsigned int i=0; i<numberOfBiases(); i++) {
				getBiasPntrs()[i]->updateReweightFactor();
			}
		}
		
		//
		if(isBiasOutputActive() && getIterationCounter()%getBiasOutputStride()==0) {
			writeBiasOutputFiles();
		}
		if(isFesOutputActive() && getIterationCounter()%getFesOutputStride()==0) {
			writeFesOutputFiles();
		}
		if(isFesProjOutputActive() && getIterationCounter()%getFesProjOutputStride()==0) {
			writeFesProjOutputFiles();
		}
		if(isTargetDistOutputActive() && getIterationCounter()%getTargetDistOutputStride()==0) {
			writeTargetDistOutputFiles();
		}
		if(isTargetDistProjOutputActive() && getIterationCounter()%getTargetDistProjOutputStride()==0) {
			writeTargetDistProjOutputFiles();
		}
	}
}

// training the parameter of discriminator
float Opt_TALOS::update_Dw(const std::vector<float>& all_input_rc,std::vector<std::vector<float>>& vec_fw)
{
	ComputationGraph cg;
	Dim xs_dim({target_dim},batch_size);
	Dim xt_dim({target_dim},ntarget);
	Dim pt_dim({1},ntarget);
	
	//~ std::random_shuffle(input_target.begin(), input_target.end());

	unsigned wsize=batch_size*target_dim;
	std::vector<float> input_sample(wsize);
	Expression x_sample=input(cg,xs_dim,&input_sample);
	Expression x_target=input(cg,xt_dim,&input_target);
	Expression p_target=input(cg,pt_dim,&target_dis);

	Expression y_sample=nn_Dw.run(x_sample,cg);
	Expression y_target=nn_Dw.run(x_target,cg);
	
	Expression l_target=y_target*p_target;

	Expression loss_sample=mean_batches(y_sample);
	Expression loss_target=mean_batches(l_target);

	Expression loss_Dw;
	
	if(do_quadratic)
	{
		Expression loss_sample2 = lambda * loss_sample * loss_sample;
		loss_Dw = loss_sample - loss_target + loss_sample2;
	}
	else
		loss_Dw = loss_sample - loss_target;
	
	double dwloss=0;
	double loss;
	std::vector<std::vector<float>> vec_input_sample;
	for(unsigned i=0;i!=nepoch;++i)
	{
		for(unsigned j=0;j!=wsize;++j)
			input_sample[j]=all_input_rc[i*wsize+j];
		vec_input_sample.push_back(input_sample);
		loss = as_scalar(cg.forward(loss_Dw));
		dwloss += loss;
		cg.backward(loss_Dw);
		train_Dw->update();
		//~ std::cout<<"Loss = "<<loss<<std::endl;
		if(do_clip)
			nn_Dw.clip(clip_left,clip_right);
	}
	dwloss/=nepoch;
	
	for(unsigned i=0;i!=nepoch;++i)
	{
		input_sample=vec_input_sample[i];
		vec_fw.push_back(as_vector(cg.forward(y_sample)));
	}
	
	return dwloss;
}

// update the coeffients of the basis function
float Opt_TALOS::update_bias(const std::vector<float>& all_input_bias,const std::vector<std::vector<float>>& vec_fw)
{
	float avg_fw=0;
	if(use_full_loss)
	{
		for(unsigned i=0;i!=vec_fw.size();++i)
		{
			for(unsigned j=0;j!=vec_fw[i].size();++j)
				avg_fw+=vec_fw[i][j];
		}
		avg_fw/=tot_data_size;
	}
	
	ComputationGraph cg;
	Expression W = parameter(cg,parm_bias);
	Dim bias_dim({tot_basis},batch_size),fw_dim({1},batch_size);
	unsigned bsize=batch_size*tot_basis;
	std::vector<float> basis_batch(bsize);
	Expression x = input(cg,bias_dim,&basis_batch);
	Expression y_pred = W * x;
	std::vector<float> fw_batch;
	Expression fw = input(cg,fw_dim,&fw_batch);
	Expression loss_mean = mean_batches(fw * y_pred);
	
	Expression loss_fin;
	if(use_full_loss)
	{
		Expression loss_mean2 = mean_batches(avg_fw * y_pred);
		loss_fin = beta * (loss_mean + loss_mean2);
	}
	else
		loss_fin = beta * loss_mean;

	double vbloss=0;
	for(unsigned i=0;i!=nepoch;++i)
	{
		fw_batch=vec_fw[i];
		for(unsigned j=0;j!=bsize;++j)
			basis_batch[j]=all_input_bias[i*bsize+j];
		vbloss += as_scalar(cg.forward(loss_fin));
		cg.backward(loss_fin);
		train_bias->update();
	}
	vbloss/=nepoch;
	
	//~ new_coe=as_vector(W.value());
	return vbloss;
}

float Opt_TALOS::update_bias_and_rc(const std::vector<float>& all_input_bias,
	const std::vector<float>& all_rc_arg_sample,
	const std::vector<float>& all_p_rc_sample1,
	const std::vector<float>& all_p_rc_sample2,
	const std::vector<std::vector<float>>& vec_fw)
{
	ComputationGraph cg;
	Expression W = parameter(cg,parm_bias);
	Dim bias_dim({tot_basis},batch_size),fw_dim({1},batch_size);
	unsigned bsize=batch_size*tot_basis;
	std::vector<float> basis_batch(bsize);
	Expression x = input(cg,bias_dim,&basis_batch);
	Expression y_pred = W * x;
	std::vector<float> fw_batch;
	Expression fw = input(cg,fw_dim,&fw_batch);
	
	Expression loss_bias = beta * fw * y_pred;
	Expression loss_mean = mean_batches(loss_bias);
	
	unsigned asize=batch_size*narg;
	Dim rc_dim({narg},batch_size);
	std::vector<float> rc_arg_batch(asize);
	Expression x_r = input(cg,rc_dim,&rc_arg_batch);
	
	Dim prc_dim({target_dim},batch_size);
	std::vector<float> p_rc1_batch(batch_size);
	std::vector<float> p_rc2_batch(batch_size);
	Expression p_rc1 = input(cg,prc_dim,&p_rc1_batch);
	Expression p_rc2 = input(cg,prc_dim,&p_rc2_batch);
	
	Expression Dw_rc = parameter(cg,parm_rc_dw);
	Expression Vb_rc = parameter(cg,parm_rc_vb);
	
	Expression y_rc = (tanh( Dw_rc * x_r + Vb_rc) + 1.0)/2;
	Expression l_rc1 = dynet::log(y_rc) * p_rc1;
	Expression l_rc2 = dynet::log(1.0 - y_rc) * p_rc2;
	
	Expression loss_rc = mean_batches(l_rc1) + mean_batches(l_rc2);
	//~ Expression loss_rc = mean_batches(l_rc1 + l_rc2);
	Expression loss_fin = loss_mean - loss_rc;
	//~ Expression loss_fin = loss_mean;

	double vbloss=0;
	for(unsigned i=0;i!=nepoch;++i)
	{
		fw_batch=vec_fw[i];
		for(unsigned j=0;j!=batch_size;++j)
		{
			p_rc1_batch[j]=all_p_rc_sample1[i*batch_size+j];
			p_rc2_batch[j]=all_p_rc_sample2[i*batch_size+j];
		}
		for(unsigned j=0;j!=asize;++j)
		{
			rc_arg_batch[j]=all_rc_arg_sample[i*asize+j];
		}
		
		for(unsigned j=0;j!=bsize;++j)
		{
			basis_batch[j]=all_input_bias[i*bsize+j];
		}
		vbloss += as_scalar(cg.forward(loss_fin));
		cg.backward(loss_fin);
		train_bias->update();
	}
	vbloss/=nepoch;
	
	//~ new_coe=as_vector(W.value());
	return vbloss;
}

void Opt_TALOS::update_rc_target()
{	
	ComputationGraph cg;
	Expression W = parameter(cg,parm_rc_dw);
	Expression b = parameter(cg,parm_rc_vb);
	Dim x_dim({narg},3);
	Expression x = input(cg,x_dim,&ps_boundary);
	Expression y_pred = tanh( W * x + b);
	std::vector<float> rcp=as_vector(cg.forward(y_pred));
	
	if((rcp[0]-rcp[2])*(rcp[2]-rcp[1])<0)
		plumed_merror("The point of transition state must between the two stable states: "+
		std::to_string(rcp[0])+", "+std::to_string(rcp[1])+", "+std::to_string(rcp[2]));
	
	//~ float gmin=grid_min[0];
	//~ float gmin=grid_min[0];
	for(unsigned i=0;i!=rcp.size();++i)
	{
		plumed_massert(rcp[i]>grid_min[0],"the point of target must be larger than the GRID_MIN");
		plumed_massert(rcp[i]<grid_max[0],"the point of target must be smaller than the GRID_MAX");
	}
	
	float r1=rcp[0];
	float r2=rcp[1]; 
		
	float sigma1=fabs(rcp[0]-rcp[2])/2;
	float sigma2=fabs(rcp[1]-rcp[2])/2;
	
	float a1=1.0/(sigma1*sqrt(2.0*PI));
	float a2=2.0/(sigma2*sqrt(2.0*PI));
	float b1=-1.0/(2*sigma1*sigma1);
	float b2=-1.0/(2*sigma2*sigma2);
	
	double sum=0;
	for(unsigned i=0;i!=ntarget;++i)
	{
		float val=grid_min[0]+grid_space[0]*i;
		input_target.push_back(val);
		float valm1=val-r1;
		float p1=a1*exp(b1*valm1*valm1);
		float valm2=val-r2;
		float p2=a2*exp(b2*valm2*valm2);
		
		target_dis[i]=p1+p2;
		sum+=p1+p2;
	}
	for(unsigned i=0;i!=ntarget;++i)
		target_dis[i]/=(sum/ntarget);
}

template<class T>
bool Opt_TALOS::parseVectorAuto(const std::string& keyword, std::vector<T>& values, unsigned num)
{
	plumed_massert(num>0,"the adjust number must be larger than 0!");
	values.resize(0);
	parseVector(keyword,values);
	if(values.size()!=num)
	{
		if(values.size()==1)
		{
			for(unsigned i=1;i!=num;++i)
				values.push_back(values[0]);
		}
		else
			plumed_merror("The number of "+keyword+" must be equal to the number of arguments!");
	}
	return true;
}

template<class T>
bool Opt_TALOS::parseVectorAuto(const std::string& keyword, std::vector<T>& values, unsigned num,const T& def_value)
{
	plumed_massert(num>0,"the adjust number must be larger than 0!");
	values.resize(0);
	parseVector(keyword,values);
	
	if(values.size()!=num)
	{
		if(values.size()==0)
		{
			for(unsigned i=0;i!=num;++i)
				values.push_back(def_value);
		}
		else if(values.size()==1)
		{
			for(unsigned i=1;i!=num;++i)
				values.push_back(values[0]);
		}
		else
			plumed_merror("The number of "+keyword+" must be equal to the number of arguments!");
	}
	return true;
}


}
}

#endif

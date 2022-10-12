#include "lietorch/pose.h"
#include <fstream>

using namespace lietorch;
using namespace std;
using namespace torch;

#define N 40000

static const Tensor skewMultiplier = torch::tensor({{{ 0, 0, 0}, { 0, 0,-1}, { 0, 1, 0}},
										   {{ 0, 0, 1}, { 0, 0, 0}, {-1, 0, 0}},
										   {{ 0,-1, 0}, { 1, 0, 0}, { 0, 0, 0}}}, kFloat);

static const Tensor rotationMultiplier = torch::tensor({{{ 1, 0, 0, 0}, { 0,-1, 0, 0}, { 0, 0,-1, 0}, { 0, 0, 0, 1}},
											  {{ 0, 2, 0, 0}, { 0, 0, 0, 0}, { 0, 0, 0, 0}, { 0, 0,-2, 0}},
											  {{ 0, 0, 2, 0}, { 0, 0, 0, 0}, { 0, 0, 0, 0}, { 0, 2, 0, 0}},
											  {{ 0, 2, 0, 0}, { 0, 0, 0, 0}, { 0, 0, 0, 0}, { 0, 0, 2, 0}},
											  {{-1, 0, 0, 0}, { 0, 1, 0, 0}, { 0, 0,-1, 0}, { 0, 0, 0, 1}},
											  {{ 0, 0, 0, 0}, { 0, 0, 2, 0}, { 0, 0, 0, 0}, {-2, 0, 0, 0}},
											  {{ 0, 0, 2, 0}, { 0, 0, 0, 0}, { 0, 0, 0, 0}, { 0,-2, 0, 0}},
											  {{ 0, 0, 0, 0}, { 0, 0, 2, 0}, { 0, 0, 0, 0}, { 2, 0, 0, 0}},
											  {{-1, 0, 0, 0}, { 0,-1, 0, 0}, { 0, 0, 1, 0}, { 0, 0, 0, 1}}}, kFloat);

void tocsv (const Tensor &v, const std::string &name) {
	ofstream file;
	file.open ("/home/nicola/test_lie_" + name + ".csv");
	for (int i = 0; i < v.size(0); i++) {
		
		if (v.sizes ().size () > 1) {
			for (int j = 0; j < v.size(1); j++)
				file << v[i][j].item ().toFloat () << ((v.size(1) - j) > 1 ? ", "  : "");
			file << "\n";
			
		} else
			file << v[i].item().toFloat () << ((v.size(0) - i) > 1 ? ", " : "");
		
	}
	file.close ();
}

Tensor skew (const Tensor &v) {
	return (skewMultiplier.unsqueeze(3) *
		   v.t().unsqueeze(1).unsqueeze(1)
		   ).sum(0).permute({2,0,1});
}

Tensor actionJacobian (const Tensor &q, const Tensor &v) {
	return -q.unsqueeze(0).unsqueeze(1).matmul(rotationMultiplier)
			  .matmul(q.unsqueeze(1)
						.unsqueeze(0))
			  .squeeze().reshape({3,3})
			  .unsqueeze(0)
			  .matmul(skew(v));
}

manif::SE3<float> poseToManif (const lietorch::Pose &poseLietorch) {
	manif::SE3<float> ret;
	
	ret.coeffs ()[0] = poseLietorch.coeffs[0].item().toFloat ();
	ret.coeffs ()[1] = poseLietorch.coeffs[1].item().toFloat ();
	ret.coeffs ()[2] = poseLietorch.coeffs[2].item().toFloat ();
	ret.coeffs ()[3] = poseLietorch.coeffs[3].item().toFloat ();
	ret.coeffs ()[4] = poseLietorch.coeffs[4].item().toFloat ();
	ret.coeffs ()[5] = poseLietorch.coeffs[5].item().toFloat ();
	ret.coeffs ()[6] = poseLietorch.coeffs[6].item().toFloat ();
	
	return ret;
}


//manif::SE3Tangent<float> tangentToManif (const lietorch::Twist)

int main ()
{
	Pose pose(Position (torch::rand ({3}, kFloat)),
			 AngularVelocity (torch::rand ({3}, kFloat)).exp ());

	Pose pose2(Position (torch::rand ({3}, kFloat)),
			  AngularVelocity (torch::rand ({3}, kFloat)).exp ());
	
	Tensor v = torch::rand ({40, 3}, kFloat);
	Tensor outerGradient = torch::rand ({40, 3}, kFloat);
	manif::SE3<float> poseManif;
	
	poseManif = poseToManif (pose);

	Quaternion a = Quaternion(0,0,0,1);
	AngularVelocity vel(10 * torch::tensor ({0.1, 0., 0.}, kFloat));

	COUTN (a * vel.exp());
/*
	Tensor lietorchJac = ops::quaternion::actionJacobian (pose.rotation ().coeffs, v);
	COUTN(Pose (Position ({1,2,3}),Quaternion (4,5,6,7)))
	COUTN(Pose::Identity ().log ());
//COUTN(lietorchJac.matmul (outerGradient.unsqueeze (2)).squeeze ());

	tocsv (pose.coeffs, "pose");
	tocsv (v, "v");
	
	tocsv (pose.log ().coeffs, "log");
	
	*/
	return 0;
}

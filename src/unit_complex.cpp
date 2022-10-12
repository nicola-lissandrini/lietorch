#include "lietorch/unit_complex.h"
#include <complex.h>

using namespace lietorch;
using namespace torch;
using namespace std;

static const torch::Tensor zeroReal = torch::zeros({1},torch::kFloat);

static const Tensor toComplex = torch::cat ({torch::complex (torch::tensor ({1}, kFloat), torch::tensor ({0}, kFloat)),
											torch::complex (torch::tensor ({0}, kFloat), torch::tensor ({1}, kFloat))});

UnitComplex::UnitComplex (float angle):
	  UnitComplex(ComplexVelocity(angle).exp ())
{}

UnitComplex UnitComplex::inverse () const {
	return conj ();
}

ComplexVelocity UnitComplex::log () const {
	return torch::imag (coeffs.log ());
}

UnitComplex UnitComplex::compose (const UnitComplex &other) const {
	return coeffs * other.coeffs;
}

Tensor UnitComplex::dist (const UnitComplex &other, const Tensor &weights) const {
	return (minus (other)).scale (weights).norm ();
}

Tensor UnitComplex::act (const Tensor &vector) const {
	bool vectorBatch = (vector.sizes ().size () > 1);
	Tensor vectComplex = toComplex.unsqueeze(0).mm ((vectorBatch ? vector.t() : vector.unsqueeze(1)).toType (kComplexFloat)).squeeze ();
	Tensor actComplex = coeffs * vectComplex;
	Tensor ret = torch::stack ({torch::real(actComplex), torch::imag(actComplex)}, 1);

	return vectorBatch ? ret : ret.squeeze ();
}

Tensor UnitComplex::real () const {
	return torch::real(coeffs);
}

Tensor UnitComplex::imag () const {
	return torch::imag (coeffs);
}

UnitComplex UnitComplex::conj () const {
	return coeffs.conj ();
}

ComplexVelocity::ComplexVelocity (float angle):
	  ComplexVelocity (torch::tensor ({angle}, kFloat))
{}

UnitComplex ComplexVelocity::exp () const {
	return torch::complex (zeroReal, coeffs).exp ();
}

Tensor ComplexVelocity::norm () const {
	return coeffs.norm ();
}

ComplexVelocity ComplexVelocity::scale (const Tensor &other) const {
	return other * coeffs;
}








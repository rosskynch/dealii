// ---------------------------------------------------------------------
//
// Copyright (C) 2005 - 2017 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------


// compute the deviator tensor as stated in the documentation of outer_product

#include "../tests.h"
#include <deal.II/base/symmetric_tensor.h>


template <int dim>
void test ()
{
  deallog << "dim=" << dim << std::endl;

  const SymmetricTensor<4,dim>
  T = (identity_tensor<dim>()
       - 1./dim * outer_product(unit_symmetric_tensor<dim>(),
                                unit_symmetric_tensor<dim>()));

  AssertThrow ((T-deviator_tensor<dim>()).norm()
               <= 1e-15*T.norm(), ExcInternalError());
}




int main ()
{
  std::ofstream logfile("output");
  deallog << std::setprecision(3);
  deallog.attach(logfile);

  test<1> ();
  test<2> ();
  test<3> ();

  deallog << "OK" << std::endl;
}

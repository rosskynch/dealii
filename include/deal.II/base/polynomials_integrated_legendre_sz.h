// ---------------------------------------------------------------------
//
// Copyright (C) 2015 - 2016 by the deal.II authors
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

#ifndef dealii__polynomials_nedelec_sz_h
#define dealii__polynomials_nedelec_sz_h

#include <deal.II/base/polynomial.h>
#include <deal.II/base/point.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/thread_management.h>

#include <cmath>
#include <algorithm>
#include <limits>

DEAL_II_NAMESPACE_OPEN

// Written using original deal.II legendre class as a base, but with the coefficents adjusted
// so that the recursive formula is for the integrated legendre polynomials described
// in the PhD thesis of Sabine Zaglmayer.
//
// L0 = -1  (added so that it can be generated recursively from 0)
// L1 = x
// L2 = 0.5*(x^2 - 1)
// (n+1)Lnp1 = (2n-1)xLn - (n-2)Lnm1.
//
// However, it is possible to generate them directly from the legendre polynomials:
//
// (2n-1)Ln = ln - lnm2
//
class integratedLegendreSZ : public Polynomials::Polynomial<double>
{
public:
  integratedLegendreSZ (unsigned int p);
  
  static
  std::vector<Polynomials::Polynomial<double> >
  generate_complete_basis (const unsigned int degree); 
  
private:
  static std::vector<std_cxx11::shared_ptr<const std::vector<double> > > shifted_coefficients;
  
  static std::vector<std_cxx11::shared_ptr<const std::vector<double> > > recursive_coefficients;
  
  static void compute_coefficients (const unsigned int p);
  
  static const std::vector<double> &
  get_coefficients (const unsigned int k);
};

DEAL_II_NAMESPACE_CLOSE

#endif
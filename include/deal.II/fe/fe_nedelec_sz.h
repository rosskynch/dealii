// ---------------------------------------------------------------------
//
// Copyright (C) 2015 - 2017 by the deal.II authors
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

#ifndef dealii__fe_nedelec_sz_h
#define dealii__fe_nedelec_sz_h

#include <deal.II/base/derivative_form.h>
#include <deal.II/base/qprojector.h>
#include <deal.II/base/polynomials_integrated_legendre_sz.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>

DEAL_II_NAMESPACE_OPEN

template <int dim>
class FE_NedelecSZ : public FiniteElement<dim,dim>
// Note: spacedim=dim
{
public:
  // Constructor
  FE_NedelecSZ (const unsigned int degree);

  // for documentation, see the FiniteElement base class
  virtual
  UpdateFlags
  requires_update_flags (const UpdateFlags update_flags) const;
  
  virtual std::string get_name () const;
  
  virtual std::unique_ptr<FiniteElement<dim,dim> > clone() const;
  
  // This element is vector-valued so throw an exception:
  virtual double shape_value (const unsigned int i,
                              const Point<dim> &p) const;

  // Not implemented yet.
  virtual double shape_value_component (const unsigned int i,
                                        const Point<dim> &p,
                                        const unsigned int component) const;

  // This element is vector-valued so throw an exception:
  virtual Tensor<1,dim> shape_grad (const unsigned int  i,
                                    const Point<dim>   &p) const;
  
  // Not implemented yet.
  virtual Tensor<1,dim> shape_grad_component (const unsigned int i,
                                              const Point<dim> &p,
                                              const unsigned int component) const;

  // This element is vector-valued so throw an exception:
  virtual Tensor<2,dim> shape_grad_grad (const unsigned int  i,
                                         const Point<dim> &p) const;

  virtual Tensor<2,dim> shape_grad_grad_component (const unsigned int i,
                                                   const Point<dim> &p,
                                                   const unsigned int component) const;

  /**
   * Given <tt>flags</tt>, determines the values which must be computed only
   * for the reference cell. Make sure, that #mapping_type is set by the
   * derived class, such that this function can operate correctly.
   */
  UpdateFlags update_once (const UpdateFlags flags) const;
  /**
   * Given <tt>flags</tt>, determines the values which must be computed in
   * each cell cell. Make sure, that #mapping_type is set by the derived
   * class, such that this function can operate correctly.
   */
  UpdateFlags update_each (const UpdateFlags flags) const;
 
protected:
  /**
   * The mapping type to be used to map shape functions from the reference
   * cell to the mesh cell.
   */
  MappingType mapping_type;

  virtual
  typename FiniteElement<dim,dim>::InternalDataBase *
  get_data (const UpdateFlags                                                    update_flags,
            const Mapping<dim,dim>                                              &mapping,
            const Quadrature<dim>                                               &quadrature,
            dealii::internal::FEValues::FiniteElementRelatedData<dim, dim> &/*output_data*/) const;

  virtual void
  fill_fe_values (const typename Triangulation<dim,dim>::cell_iterator           &cell,
                  const CellSimilarity::Similarity                                     cell_similarity,
                  const Quadrature<dim>                                               &quadrature,
                  const Mapping<dim,dim>                                         &mapping,
                  const typename Mapping<dim,dim>::InternalDataBase              &mapping_internal,
                  const dealii::internal::FEValues::MappingRelatedData<dim, dim> &mapping_data,
                  const typename FiniteElement<dim,dim>::InternalDataBase        &fedata,
                  dealii::internal::FEValues::FiniteElementRelatedData<dim, dim> &data) const;


  virtual void
  fill_fe_face_values (const typename Triangulation<dim,dim>::cell_iterator           &cell,
                       const unsigned int                                              face_no,
                       const Quadrature<dim-1>                                        &quadrature,
                       const Mapping<dim,dim>                                         &mapping,
                       const typename Mapping<dim,dim>::InternalDataBase              &mapping_internal,
                       const dealii::internal::FEValues::MappingRelatedData<dim, dim> &mapping_data,
                       const typename FiniteElement<dim,dim>::InternalDataBase        &fedata,
                       dealii::internal::FEValues::FiniteElementRelatedData<dim, dim> &data) const;

  virtual void
  fill_fe_subface_values (const typename Triangulation<dim,dim>::cell_iterator           &cell,
                          const unsigned int                                              face_no,
                          const unsigned int                                              sub_no,
                          const Quadrature<dim-1>                                        &quadrature,
                          const Mapping<dim,dim>                                         &mapping,
                          const typename Mapping<dim,dim>::InternalDataBase              &mapping_internal,
                          const dealii::internal::FEValues::MappingRelatedData<dim, dim> &mapping_data,
                          const typename FiniteElement<dim,dim>::InternalDataBase        &fedata,
                          dealii::internal::FEValues::FiniteElementRelatedData<dim, dim> &data) const;
                          
                          
  // DERVIED INTERNAL DATA - CAN BE USED TO STORE PRECOMPUTED INFO
  // E.G. Once dim is known we can compute sigma, lambda, and combinations.
  class InternalData : public FiniteElement<dim,dim>::InternalDataBase
  {
  public:
    // Storage for shape functions on the reference element
    // We only pre-compute those cell-based DoFs, as the edge-based
    // dofs depend on the choice of cell.
    mutable std::vector<std::vector<Tensor<1,dim> > > shape_values;
    
    mutable std::vector<std::vector<DerivativeForm<1, dim, dim> > > shape_grads;
    
    // TODO: remove the redundant parts, will need to update the 2D case when this is removed.
    // - sigma - these won't be reused.
    // - lambda - won't be reused.
    // - lambda_ipj = lambda[i] + lambda[j] - no need, can go straight to the edge_lambda_*.
    // - edgeDoF_to_poly - should remove the use of polyspace and just use the 1d polynomial for everything.
    
    
    // Sigma_imj = sigma_i - sigma_j gives a parameterisation of an edge
    // connected by vertices i and j. The values, gradients, non-zero
    // component and sign of the co-efficient of x, y or z are stored.
    // The orientation of edges and faces can then be handled using sigma_imj.
    std::vector<std::vector<std::vector<double> > > sigma_imj_values;
    std::vector<std::vector<std::vector<double> > > sigma_imj_grads;

    // Storage for "standard" edges:
    // On edge, E_m = [e_{1}^{m}, e_{2}^{m}].
    // edge_sigma[m][q] = sigma[e2][q] - sigma[e1][q]
    // edge_lambda_values[m][q] = lambda[e1][q] + lambda[e1][q]
    // The edge sigma values will be adjusted for the orientation of the edges of the physical cell.
    // The edge lambda values do not change with orientation.
    // Note that the gradient components of edge_sigma are constant,
    // so we don't need the values at every quad point, whereas they are
    // needed for the lambda gradient.    
    std::vector<std::vector<double> > edge_sigma_values;
    std::vector<std::vector<double> > edge_sigma_grads;
    
    std::vector<std::vector<double> > edge_lambda_values;
    std::vector<unsigned int> edge_lambda_component;
    
    // In 2D, the lambda grads are constant, but not in 3D.
    std::vector<std::vector<double> > edge_lambda_grads_2d;   
    // There are non-zero second derivatives for lambda,
    // but they are constant across the cell.
    std::vector<std::vector<std::vector<double> > > edge_lambda_grads_3d;
    std::vector<std::vector<std::vector<double> > > edge_lambda_gradgrads_3d;
    
    // TODO: Add face parameterisation storage for xi/eta.
    std::vector<std::vector<double> > face_lambda_values;
    std::vector<std::vector<double> > face_lambda_grads;
  };
  
private:
  
  static std::vector<unsigned int> get_dpo_vector (unsigned int degree);
  
  std::vector<Polynomials::Polynomial<double> > IntegratedLegendrePolynomials;
  
  void  create_polynomials (const unsigned int degree);

  // returns the number of dofs per vertex/edge/face/cell
  std::vector<unsigned int> get_n_pols(unsigned int degree);
  
  // returns the number of polynomials in the basis set.
  unsigned int compute_n_pols (unsigned int degree);
  
  // Calculates cell-dependent edge-based shape functions.
  void fill_edge_values(const typename Triangulation<dim,dim>::cell_iterator &cell,  
                        const Quadrature<dim>                                &quadrature,
                        const InternalData                                   &fedata) const;
  
  // Calculates cell-dependent face-based shape functions.
  void fill_face_values(const typename Triangulation<dim,dim>::cell_iterator &cell,  
                        const Quadrature<dim>                                &quadrature,
                        const InternalData                                   &fedata) const;
};

DEAL_II_NAMESPACE_CLOSE

#endif
  

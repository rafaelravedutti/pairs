#pragma once

//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <cstddef>

namespace pairs {


//=================================================================================================
//
//  MATHEMATICAL TRAIT
//
//=================================================================================================

//*************************************************************************************************
/*!\class MathTrait
 * \brief Base template for the MathTrait class.
 * \ingroup math
 *
 * \section mathtrait_general General
 *
 * The MathTrait class template offers the possibility to select the resulting data type
 * of a generic mathematical operation. In case of operations between built-in data types,
 * the MathTrait class defines the more significant data type as the resulting data type.
 * For this selection, signed data types are given a higher significance. In case of
 * operations involving user-defined data types, the MathTrait template specifies the
 * resulting data type of this operation.\n
 * Specifying the resulting data type for a specific operation is done by specializing
 * the MathTrait template for this particular type combination. In case a certain type
 * combination is not defined in a MathTrait specialization, the base template is selected,
 * which defines no resulting types and therefore stops the compilation process. Each
 * specialization defines the data types \a HighType that represents the high-order data
 * type of the two given data types and \a LowType that represents the low-order data type.
 * Additionally, each specialization defines the types \a AddType, \a SubType, \a MultType
 * and \a DivType, that represent the type of the resulting data type of the corresponding
 * mathematical operation. The following example shows the specialization for operations
 * between the double and the integer type:

   \code
   template<>
   struct MathTrait< double, int >
   {
      typedef double  HighType;
      typedef int     LowType;
      typedef double  AddType;
      typedef double  SubType;
      typedef double  MultType;
      typedef double  DivType;
   };
   \endcode

 * Per default, the MathTrait template provides specializations for the following built-in
 * data types:
 *
 * <ul>
 *    <li>integers</li>
 *    <ul>
 *       <li>unsigned char, signed char, char, wchar_t</li>
 *       <li>unsigned short, short</li>
 *       <li>unsigned int, int</li>
 *       <li>unsigned long, long</li>
 *       <li>std::size_t, std::ptrdiff_t (for certain 64-bit compilers)</li>
 *    </ul>
 *    <li>floating points</li>
 *    <ul>
 *       <li>float</li>
 *       <li>double</li>
 *       <li>long double</li>
 *    </ul>
 * </ul>
 *
 *
 * \n \section specializations Creating custom specializations
 *
 * It is possible to specialize the MathTrait template for additional user-defined data types.
 * However, it is possible that a specific mathematical operation is invalid for the particular
 * type combination. In this case, the INVALID_NUMERICAL_TYPE can be used to fill the missing
 * type definition. The INVALID_NUMERICAL_TYPE represents the resulting data type of an invalid
 * numerical operation. It is left undefined to stop the compilation process in case it is
 * instantiated. The following example shows the specialization of the MathTrait template for
 * Matrix3 and Vector3. In this case, only the multiplication between the matrix and the vector
 * is a valid numerical operation. Therefore for all other types the INVALID_NUMERICAL_TYPE is
 * used.

   \code
   template< typename T1, typename T2 >
   struct MathTrait< Matrix3<T1>, Vector3<T2> >
   {
      typedef INVALID_NUMERICAL_TYPE                          HighType;  // Invalid, no common high data type
      typedef INVALID_NUMERICAL_TYPE                          LowType;   // Invalid, no common low data type
      typedef INVALID_NUMERICAL_TYPE                          AddType;   // Invalid, cannot add a matrix and a vector
      typedef INVALID_NUMERICAL_TYPE                          SubType;   // Invalid, cannot subtract a vector from a matrix
      typedef Vector3< typename MathTrait<T1,T2>::MultType >  MultType;  // Multiplication between a matrix and a vector
      typedef INVALID_NUMERICAL_TYPE                          DivType;   // Invalid, cannot divide a matrix by a vector
   };
   \endcode

 * \n \section mathtrait_examples Examples
 *
 * The following example demonstrates the use of the MathTrait template, where depending on
 * the two given data types the resulting data type is selected:

   \code
   template< typename T1, typename T2 >    // The two generic types
   typename MathTrait<T1,T2>::HighType     // The resulting generic return type
   add( T1 t1, T2 t2 )                     //
   {                                       // The function 'add' returns the sum
      return t1 + t2;                      // of the two given values
   }                                       //
   \endcode

 * Additionally, the specializations of the MathTrait template enable arithmetic operations
 * between any combination of the supported data types:

   \code
   typedef Vector3< Matrix3< float  > >  VectorOfMatrices;  // Vector of single-precision matrices
   typedef Vector3< Vector3  < double > >  VectorOfVectors;   // Vector of double-precision vectors
   typedef Vector3< double >                   VectorOfScalars;   // Vector of double-precision scalars

   VectorOfMatrices vm;  // Setup of a vector of matrices
   VectorOfVectors  vv;  // Setup of a vector of vectors

   // Calculation of the scalar product between the two vectors. The resulting data type
   // is a plain 3-dimensional vector of scalar values of type double.
   VectorOfScalars res = vm * vv;
   \endcode
 */
//*************************************************************************************************

//strange but needed for compatibility reasons with visual studio compiler
//backward compatibility to old PAIRS code
template< typename T1, typename T2 >
struct MathTrait
{
   using HighType = T1;
   using LowType = T2;
   using High = T1;
   using Low = T2;
};

template< typename T>
struct MathTrait< T, T >
{
   using HighType = T;
   using LowType = T;
   using High = T;
   using Low = T;
};


//=================================================================================================
//
//  MATHTRAIT SPECIALIZATION MACRO
//
//=================================================================================================

//*************************************************************************************************
/*! \cond internal */
/*!\brief Macro for the creation of MathTrait specializations for the built-in data types.
 * \ingroup math
 *
 * This macro is used for the setup of the MathTrait specializations for the built-in data
 * types.
 */
#define PAIRS_CREATE_MATHTRAIT_SPECIALIZATION(T1,T2,HIGH,LOW) \
   template<> \
   struct MathTrait< T1, T2 > \
   { \
      typedef HIGH  HighType; \
      typedef LOW   LowType;  \
      typedef HIGH  High;     \
      typedef LOW   Low;      \
      typedef HIGH  AddType;  \
      typedef HIGH  SubType;  \
      typedef HIGH  MultType; \
      typedef HIGH  DivType;  \
   }
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  UNSIGNED CHAR SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond internal */
//                                  Type 1          Type 2          High type       Low type
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned char , unsigned char , unsigned char , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned char , char          , char          , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned char , signed char   , signed char   , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned char , wchar_t       , wchar_t       , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned char , unsigned short, unsigned short, unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned char , short         , short         , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned char , unsigned int  , unsigned int  , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned char , int           , int           , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned char , unsigned long , unsigned long , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned char , long          , long          , unsigned char  );
#if defined(_WIN64)
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned char , std::size_t   , std::size_t   , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned char , std::ptrdiff_t, std::ptrdiff_t, unsigned char  );
#endif
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned char , float         , float         , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned char , double        , double        , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned char , long double   , long double   , unsigned char  );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  CHAR SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond internal */
//                                  Type 1          Type 2          High type       Low type
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( char          , unsigned char , char          , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( char          , char          , char          , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( char          , signed char   , signed char   , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( char          , wchar_t       , wchar_t       , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( char          , unsigned short, unsigned short, char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( char          , short         , short         , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( char          , unsigned int  , unsigned int  , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( char          , int           , int           , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( char          , unsigned long , unsigned long , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( char          , long          , long          , char           );
#if defined(_WIN64)
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( char          , std::size_t   , std::size_t   , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( char          , std::ptrdiff_t, std::ptrdiff_t, char           );
#endif
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( char          , float         , float         , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( char          , double        , double        , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( char          , long double   , long double   , char           );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SIGNED CHAR SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond internal */
//                                  Type 1          Type 2          High type       Low type
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( signed char   , unsigned char , signed char   , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( signed char   , char          , signed char   , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( signed char   , signed char   , signed char   , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( signed char   , wchar_t       , wchar_t       , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( signed char   , unsigned short, unsigned short, signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( signed char   , short         , short         , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( signed char   , unsigned int  , unsigned int  , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( signed char   , int           , int           , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( signed char   , unsigned long , unsigned long , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( signed char   , long          , long          , signed char    );
#if defined(_WIN64)
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( signed char   , std::size_t   , std::size_t   , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( signed char   , std::ptrdiff_t, std::ptrdiff_t, signed char    );
#endif
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( signed char   , float         , float         , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( signed char   , double        , double        , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( signed char   , long double   , long double   , signed char    );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  WCHAR_T SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond internal */
//                                  Type 1          Type 2          High type       Low type
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( wchar_t       , unsigned char , wchar_t       , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( wchar_t       , char          , wchar_t       , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( wchar_t       , signed char   , wchar_t       , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( wchar_t       , wchar_t       , wchar_t       , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( wchar_t       , unsigned short, unsigned short, wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( wchar_t       , short         , short         , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( wchar_t       , unsigned int  , unsigned int  , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( wchar_t       , int           , int           , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( wchar_t       , unsigned long , unsigned long , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( wchar_t       , long          , long          , wchar_t        );
#if defined(_WIN64)
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( wchar_t       , std::size_t   , std::size_t   , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( wchar_t       , std::ptrdiff_t, std::ptrdiff_t, wchar_t        );
#endif
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( wchar_t       , float         , float         , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( wchar_t       , double        , double        , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( wchar_t       , long double   , long double   , wchar_t        );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  UNSIGNED SHORT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond internal */
//                                  Type 1          Type 2          High type       Low type
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned short, unsigned char , unsigned short, unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned short, char          , unsigned short, char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned short, signed char   , unsigned short, signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned short, wchar_t       , unsigned short, wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned short, unsigned short, unsigned short, unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned short, short         , short         , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned short, unsigned int  , unsigned int  , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned short, int           , int           , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned short, unsigned long , unsigned long , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned short, long          , long          , unsigned short );
#if defined(_WIN64)
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned short, std::size_t   , std::size_t   , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned short, std::ptrdiff_t, std::ptrdiff_t, unsigned short );
#endif
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned short, float         , float         , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned short, double        , double        , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned short, long double   , long double   , unsigned short );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SHORT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond internal */
//                                  Type 1          Type 2          High type       Low type
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( short         , unsigned char , short         , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( short         , char          , short         , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( short         , signed char   , short         , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( short         , wchar_t       , short         , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( short         , unsigned short, short         , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( short         , short         , short         , short          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( short         , unsigned int  , unsigned int  , short          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( short         , int           , int           , short          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( short         , unsigned long , unsigned long , short          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( short         , long          , long          , short          );
#if defined(_WIN64)
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( short         , std::size_t   , std::size_t   , short          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( short         , std::ptrdiff_t, std::ptrdiff_t, short          );
#endif
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( short         , float         , float         , short          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( short         , double        , double        , short          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( short         , long double   , long double   , short          );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  UNSIGNED INT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond internal */
//                                  Type 1          Type 2          High type       Low type
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned int  , unsigned char , unsigned int  , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned int  , char          , unsigned int  , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned int  , signed char   , unsigned int  , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned int  , wchar_t       , unsigned int  , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned int  , unsigned short, unsigned int  , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned int  , short         , unsigned int  , short          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned int  , unsigned int  , unsigned int  , unsigned int   );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned int  , int           , int           , unsigned int   );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned int  , unsigned long , unsigned long , unsigned int   );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned int  , long          , long          , unsigned int   );
#if defined(_WIN64)
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned int  , std::size_t   , std::size_t   , unsigned int   );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned int  , std::ptrdiff_t, std::ptrdiff_t, unsigned int   );
#endif
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned int  , float         , float         , unsigned int   );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned int  , double        , double        , unsigned int   );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned int  , long double   , long double   , unsigned int   );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  INT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond internal */
//                                  Type 1          Type 2          High type       Low type
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( int           , unsigned char , int           , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( int           , char          , int           , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( int           , signed char   , int           , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( int           , wchar_t       , int           , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( int           , unsigned short, int           , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( int           , short         , int           , short          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( int           , unsigned int  , int           , unsigned int   );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( int           , int           , int           , int            );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( int           , unsigned long , unsigned long , int            );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( int           , long          , long          , int            );
#if defined(_WIN64)
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( int           , std::size_t   , std::size_t   , int            );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( int           , std::ptrdiff_t, std::ptrdiff_t, int            );
#endif
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( int           , float         , float         , int            );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( int           , double        , double        , int            );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( int           , long double   , long double   , int            );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  UNSIGNED LONG SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond internal */
//                                  Type 1          Type 2          High type       Low type
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned long , unsigned char , unsigned long , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned long , char          , unsigned long , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned long , signed char   , unsigned long , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned long , wchar_t       , unsigned long , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned long , unsigned short, unsigned long , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned long , short         , unsigned long , short          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned long , unsigned int  , unsigned long , unsigned int   );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned long , int           , unsigned long , int            );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned long , unsigned long , unsigned long , unsigned long  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned long , long          , long          , unsigned long  );
#if defined(_WIN64)
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned long , std::size_t   , std::size_t   , unsigned long  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned long , std::ptrdiff_t, std::ptrdiff_t, unsigned long  );
#endif
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned long , float         , float         , unsigned long  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned long , double        , double        , unsigned long  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( unsigned long , long double   , long double   , unsigned long  );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LONG SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond internal */
//                                  Type 1          Type 2          High type       Low type
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long          , unsigned char , long          , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long          , char          , long          , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long          , signed char   , long          , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long          , wchar_t       , long          , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long          , unsigned short, long          , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long          , short         , long          , short          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long          , unsigned int  , long          , unsigned int   );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long          , int           , long          , int            );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long          , unsigned long , long          , unsigned long  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long          , long          , long          , long           );
#if defined(_WIN64)
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long          , std::size_t   , std::size_t   , long           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long          , std::ptrdiff_t, std::ptrdiff_t, long           );
#endif
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long          , float         , float         , long           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long          , double        , double        , long           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long          , long double   , long double   , long           );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  SIZE_T SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
#if defined(_WIN64)
/*! \cond internal */
//                                  Type 1          Type 2          High type       Low type
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( std::size_t   , unsigned char , std::size_t   , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( std::size_t   , char          , std::size_t   , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( std::size_t   , signed char   , std::size_t   , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( std::size_t   , wchar_t       , std::size_t   , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( std::size_t   , unsigned short, std::size_t   , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( std::size_t   , short         , std::size_t   , short          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( std::size_t   , unsigned int  , std::size_t   , unsigned int   );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( std::size_t   , int           , std::size_t   , int            );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( std::size_t   , unsigned long , std::size_t   , unsigned long  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( std::size_t   , long          , std::size_t   , long           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( std::size_t   , std::size_t   , std::size_t   , std::size_t    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( std::size_t   , float         , float         , std::size_t    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( std::size_t   , double        , double        , std::size_t    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( std::size_t   , long double   , long double   , std::size_t    );
/*! \endcond */
#endif
//*************************************************************************************************




//=================================================================================================
//
//  FLOAT SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond internal */
//                                  Type 1          Type 2          High type       Low type
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( float         , unsigned char , float         , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( float         , char          , float         , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( float         , signed char   , float         , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( float         , wchar_t       , float         , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( float         , unsigned short, float         , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( float         , short         , float         , short          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( float         , unsigned int  , float         , unsigned int   );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( float         , int           , float         , int            );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( float         , unsigned long , float         , unsigned long  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( float         , long          , float         , long           );
#if defined(_WIN64)
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( float         , std::size_t   , float         , std::size_t    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( float         , std::ptrdiff_t, float         , std::ptrdiff_t );
#endif
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( float         , float         , float         , float          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( float         , double        , double        , float          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( float         , long double   , long double   , float          );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  DOUBLE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond internal */
//                                  Type 1          Type 2          High type       Low type
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( double        , unsigned char , double        , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( double        , char          , double        , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( double        , signed char   , double        , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( double        , wchar_t       , double        , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( double        , unsigned short, double        , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( double        , short         , double        , short          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( double        , unsigned int  , double        , unsigned int   );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( double        , int           , double        , int            );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( double        , unsigned long , double        , unsigned long  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( double        , long          , double        , long           );
#if defined(_WIN64)
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( double        , std::size_t   , double        , std::size_t    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( double        , std::ptrdiff_t, double        , std::ptrdiff_t );
#endif
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( double        , float         , double        , float          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( double        , double        , double        , double         );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( double        , long double   , long double   , double         );
/*! \endcond */
//*************************************************************************************************




//=================================================================================================
//
//  LONG DOUBLE SPECIALIZATIONS
//
//=================================================================================================

//*************************************************************************************************
/*! \cond internal */
//                                  Type 1          Type 2          High type       Low type
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long double   , unsigned char , long double   , unsigned char  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long double   , char          , long double   , char           );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long double   , signed char   , long double   , signed char    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long double   , wchar_t       , long double   , wchar_t        );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long double   , unsigned short, long double   , unsigned short );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long double   , short         , long double   , short          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long double   , unsigned int  , long double   , unsigned int   );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long double   , int           , long double   , int            );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long double   , unsigned long , long double   , unsigned long  );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long double   , long          , long double   , long           );
#if defined(_WIN64)
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long double   , std::size_t   , long double   , std::size_t    );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long double   , std::ptrdiff_t, long double   , std::ptrdiff_t );
#endif
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long double   , float         , long double   , float          );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long double   , double        , long double   , double         );
PAIRS_CREATE_MATHTRAIT_SPECIALIZATION( long double   , long double   , long double   , long double    );
/*! \endcond */
//*************************************************************************************************

#undef PAIRS_CREATE_MATHTRAIT_SPECIALIZATION

}

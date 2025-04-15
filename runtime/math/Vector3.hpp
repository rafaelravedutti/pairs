#pragma once
#include <iostream>

#include "../pairs_common.hpp"
#include "MathTrait.h"


namespace pairs {

#define HIGH typename MathTrait<Type,Other>::High

template< typename Type >
class Vector3 {
private:
   template< typename Other > friend class Vector3;
   Type v_[3] = {Type(), Type(), Type()};

public:
   // If the constructor is called from device, v_ is automatically allocated on 
   // device because it's a static array embeded in the object itself 
   //**Constructors*****************************************************************************************************
   Vector3() = default;

   PAIRS_ATTR_HOST_DEVICE inline Vector3( Type init )
   {
      v_[0] = v_[1] = v_[2] = init;
   }

   template< typename Other >
   PAIRS_ATTR_HOST_DEVICE inline Vector3( Other init )
   {
      v_[0] = v_[1] = v_[2] = static_cast<Type>(init);
   }

   PAIRS_ATTR_HOST_DEVICE inline Vector3( Type x, Type y, Type z ) {
      v_[0] = x;
      v_[1] = y;
      v_[2] = z;
   }

   PAIRS_ATTR_HOST_DEVICE inline Vector3( const Type* init )
   {
      v_[0] = init[0];
      v_[1] = init[1];
      v_[2] = init[2];
   }

   template< typename Other >
   PAIRS_ATTR_HOST_DEVICE inline Vector3( const Vector3<Other>& v )
   {
      v_[0] = static_cast<Type>( v.v_[0] );
      v_[1] = static_cast<Type>( v.v_[1] );
      v_[2] = static_cast<Type>( v.v_[2] );
   }

   //**Operators********************************************************************************************************
   PAIRS_ATTR_HOST_DEVICE inline Type& operator[]( int index ) { 
      return v_[index]; 
   }

   PAIRS_ATTR_HOST_DEVICE inline const Type& operator[] ( int index ) const { 
      return v_[index]; 
   }

   //**Arithmetic operators*********************************************************************************************
   PAIRS_ATTR_HOST_DEVICE inline Vector3<Type> operator-() const
   {
      return Vector3( -v_[0], -v_[1], -v_[2] );
   }

   template< typename Other >
   PAIRS_ATTR_HOST_DEVICE inline Vector3<HIGH> operator+( const Vector3<Other>& rhs ) const{
      return Vector3<HIGH>( v_[0]+static_cast<Type>(rhs.v_[0]), v_[1]+static_cast<Type>(rhs.v_[1]), v_[2]+static_cast<Type>(rhs.v_[2]) );
   }
   
   template< typename Other >
   PAIRS_ATTR_HOST_DEVICE inline Vector3<HIGH> operator-( const Vector3<Other>& rhs ) const{
      return Vector3<HIGH>( v_[0]-static_cast<Type>(rhs.v_[0]), v_[1]-static_cast<Type>(rhs.v_[1]), v_[2]-static_cast<Type>(rhs.v_[2]) );
   }

   template< typename Other >
   PAIRS_ATTR_HOST_DEVICE inline Vector3<HIGH> operator*( const Vector3<Other>& rhs ) const{
      return Vector3<HIGH>( v_[0]*static_cast<Type>(rhs.v_[0]), v_[1]*static_cast<Type>(rhs.v_[1]), v_[2]*static_cast<Type>(rhs.v_[2]) );
   }

   template< typename Other >
   PAIRS_ATTR_HOST_DEVICE inline Vector3<HIGH> operator*( Other rhs ) const{
      return Vector3<HIGH>( v_[0]*static_cast<Type>(rhs), v_[1]*static_cast<Type>(rhs), v_[2]*static_cast<Type>(rhs) );
   }
};


template< typename Type, typename Other >
PAIRS_ATTR_HOST_DEVICE inline Vector3<HIGH> operator*( Other scalar, const Vector3<Type>& vec )
{
   return vec * scalar;
}

template< typename Type >
PAIRS_ATTR_HOST std::ostream& operator<<( std::ostream& os, const Vector3<Type>& v )
{
   return os << "<" << v[0] << "," << v[1] << "," << v[2] << ">";
}

#undef HIGH

}

#pragma once
#include <iostream>

#include "../pairs_common.hpp"
#include "MathTrait.h"


namespace pairs {

#define HIGH typename MathTrait<Type,Other>::High

template< typename Type >
class Vector3 {
public:
   Vector3() = default;

   // If the constructor is called from device, v_ is automatically allocated on 
   // device because it's a static array embeded in the object itself 
   PAIRS_ATTR_HOST_DEVICE Vector3( Type x, Type y, Type z ) {
      v_[0] = x;
      v_[1] = y;
      v_[2] = z;
   }

   template< typename Other >
   PAIRS_ATTR_HOST_DEVICE inline Vector3<HIGH> operator+( const Vector3<Other>& rhs ) const{
      return Vector3<HIGH>( v_[0]+static_cast<Type>(rhs.v_[0]), v_[1]+static_cast<Type>(rhs.v_[1]), v_[2]+static_cast<Type>(rhs.v_[2]) );
   }

   PAIRS_ATTR_HOST_DEVICE Type& operator[]( int index ) { 
      return v_[index]; 
   }

   PAIRS_ATTR_HOST_DEVICE const Type& operator[] ( int index ) const { 
      return v_[index]; 
   }

private:
   Type v_[3] = {Type(), Type(), Type()};
};
#undef HIGH

}

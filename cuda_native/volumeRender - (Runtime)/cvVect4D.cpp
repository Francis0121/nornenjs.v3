// cvVect4D.cpp: implementation of the CVVect4D class.
//
//////////////////////////////////////////////////////////////////////

#include "stdafx.h"
#include <math.h>
#include "cvVect4D.h"

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif

CVVect4D::CVVect4D()
{
	for(int i=0; i<3 ; i++)
		m[i] = 0;

	m[3] =1;
}

CVVect4D::CVVect4D(const CVVect4D& vect)
{
	for(int i=0; i<4 ; i++)
		m[i] = vect.m[i];
}

CVVect4D::CVVect4D(double m0,double m1,double m2,double m3)
{
	m[0]=m0; m[1]=m1; m[2]=m2; m[3]=m3;
}

CVVect4D::CVVect4D(double m0,double m1,double m2)
{
	m[0]=m0; m[1]=m1; m[2]=m2; m[3]=1.;
}

CVVect4D::CVVect4D(double vec[4])
{
	m[0]=vec[0]; m[1]=vec[1]; m[2]=vec[2]; m[3]=vec[3];
}

CVVect4D::CVVect4D(CPoint pt)
{
	m[0]=pt.x; m[1]=pt.y; m[2]=0; m[3]=1;
}

CVVect4D::~CVVect4D(void)
{
}

CVVect4D CVVect4D::operator=(const CVVect4D& vect)
{
	for(int i=0; i<4 ; i++)
		m[i] = vect.m[i];

	return (*this);
}

CVVect4D CVVect4D::operator+(const CVVect4D& vect)
{
	CVVect4D resV;

	for(int i=0; i< 3; i++)
		resV.m[i] = m[i]+vect.m[i];
	
	return resV;
}

CVVect4D CVVect4D::operator-(const CVVect4D& vect)
{
	CVVect4D resV;

	for(int i=0; i< 3; i++)
		resV.m[i] = m[i]-vect.m[i];
	
	return resV;
}

/*
CVVect4D CVVect4D::operator+(const RxPoint3D<float> pt3D)
{
	CVVect4D resV;

	resV.m[0] = m[0]+pt3D.x;
	resV.m[1] = m[1]+pt3D.y;
	resV.m[2] = m[2]+pt3D.z;
	
	return resV;
}
*/

//vector³»Àû
double CVVect4D::operator*(const CVVect4D& vect)
{
	double res=0;

	for(int i=0; i< 3; i++)
		res += m[i]*vect.m[i];
	
	return res;
}

CVVect4D CVVect4D::operator*(const double& var)
{
	CVVect4D resV;

	for(int i=0; i<3; i++)
		resV.m[i] = m[i]*var;

//++chlee 
	resV.m[3] = m[3];

	return resV;
}


/*CVVect4D CVVect4D::operator*(const RxMatrix4D mat)
{
	CVVect4D resV(0,0,0,0);

	for(int i=0; i< 4; i++)
		for(int j=0; j< 4; j++) 
			resV.m[i] += m[j]*mat.m[j][i];
	
	return resV;
}*/

double CVVect4D::operator[](const int& inx) const
{
	return m[inx];
}

BOOL CVVect4D::operator==(const CVVect4D &r) const
{
	if (m[0]==r[0]&&m[1]==r[1]&&m[2]==r[2]) return TRUE;
	else return FALSE;
}

BOOL CVVect4D::operator!=(const CVVect4D &r) const
{
	return !(*this==r);
}

BOOL CVVect4D::IsParallel(CVVect4D r)
{
	double mag = Magnitude() * r.Magnitude();
	double inner = fabs((*this) * r);
	double cosTheta = (mag==0)? 0 : inner / mag;
	return (cosTheta > 0.999); // threshold
}

void CVVect4D::Normalize(void)
{
	double norm = sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);

	for(int i=0 ; i<3 ; i++)
		m[i] /= norm;
}

void CVVect4D::ToCartesianCoord(void)
{
	for(int i=0 ; i<3 ; i++)
		m[i] /= m[3];

	m[3] = 1.;
}

// sungeui start
double CVVect4D::DotProduct (const CVVect4D& vect)
{
	return m[0]*vect.m[0] + m[1]*vect.m[1] + m[2]*vect.m[2];
}

CVVect4D CVVect4D::CrossProduct (const CVVect4D& vect)
{
	CVVect4D resV;

	resV.m[0] = m[1]*vect.m[2] - m[2]*vect.m[1];
	resV.m[1] = m[2]*vect.m[0] - m[0]*vect.m[2];
	resV.m[2] = m[0]*vect.m[1] - m[1]*vect.m[0];

	return resV;
}

CVVect4D CVVect4D::operator/(const double& var)
{
	CVVect4D resV;

	for(int i=0; i<3; i++)
		resV.m[i] = m[i]/var;

//++chlee
	resV.m[3] = m[3];

	return resV;
}
double CVVect4D::Magnitude (void)
{
	double Value;

	Value = m[0] * m[0] + m[1] * m[1] + m[2] * m[2];

	return sqrt(Value);
}

CVVect4D CVVect4D::Inverse()
{
	CVVect4D resV;

	resV.m[0] = -m[0];
	resV.m[1] = -m[1];
	resV.m[2] = -m[2];

	return resV;
}

int CVVect4D::GetPrincipalAxis()
{
	if (fabs(m[0]) >= fabs(m[1])) {
		if (fabs(m[0]) >= fabs(m[2]))
			return 0;
		else
			return 2;
	}
	else {
		if (fabs(m[1]) >= fabs(m[2]))
			return 1;
		else
			return 2;
	}
}

int CVVect4D::GetLeastPrincipalAxis()
{
	if (fabs(m[0]) < fabs(m[1])) {
		if (fabs(m[0]) < fabs(m[2]))
			return 0;
		else
			return 2;
	}
	else {
		if (fabs(m[1]) < fabs(m[2]))
			return 1;
		else
			return 2;
	}

}

int CVVect4D::GetPrincipalAxisWithSign()
{
	// 1 : X_AXIS, 2 : Y_AXIS, 3 : Z_AXIS

	if (fabs(m[0]) >= fabs(m[1])-0.1e-3) {
		if (fabs(m[0]) >= fabs(m[2])-0.1e-3) {
			if (m[0] >=0)
				return 1;
			else
				return -1;
		}
		else {
			if (m[2] >= 0)
				return 3;
			else
				return -3;
		}
	}
	else {
		if (fabs(m[1]) >= fabs(m[2])-0.1e-3) {
			if (m[1] >= 0)
				return 2;
			else
				return -2;
		}
		else {
			if (m[2] >= 0)
				return 3;
			else
				return -3;
		}
	}
}

int CVVect4D::GetSecondaryAxisWithSign(int iPrincipal)
{
	// 1 : X_AXIS, 2 : Y_AXIS, 3 : Z_AXIS

	if (iPrincipal < 0) iPrincipal = -iPrincipal;

	if (iPrincipal==1) {
		if (fabs(m[1]) >= fabs(m[2])-0.1e-3) {
			if (m[1] >= 0)
				return 2;
			else
				return -2;
		}
		else {
			if (m[2] >= 0)
				return 3;
			else
				return -3;
		}
	}
	else if (iPrincipal==2) {
		if (fabs(m[0]) >= fabs(m[2])-0.1e-3) {
			if (m[0] >= 0)
				return 1;
			else
				return -1;
		}
		else {
			if (m[2] >= 0)
				return 3;
			else
				return -3;
		}
	}
	else {
		if (fabs(m[0]) >= fabs(m[1])-0.1e-3) {
			if (m[0] >= 0)
				return 1;
			else
				return -1;
		}
		else {
			if (m[1] >= 0)
				return 2;
			else
				return -2;
		}
	}
}

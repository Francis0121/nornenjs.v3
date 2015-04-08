// Matrix4D.cpp : implementation of the CSegRender class
//
#include "stdafx.h"
#include "cvMatrix4D.h"

#ifdef _DEBUG
#undef THIS_FILE
static char THIS_FILE[]=__FILE__;
#define new DEBUG_NEW
#endif

#include <math.h>

#include "ippm.h"

#define M2DET(x00, x01, x10, x11) \
  ((x00)*(x11) - (x01)*(x10))


#define M3DET(x00, x01, x02, x10, x11, x12, x20, x21, x22) \
  (  (x00) * M2DET((x11), (x12), (x21), (x22)) \
   - (x01) * M2DET((x10), (x12), (x20), (x22)) \
   + (x02) * M2DET((x10), (x11), (x20), (x21))) 

CVMatrix4D::CVMatrix4D()
{
	for(int i=0; i<4 ; i++)
		for(int j=0; j<4 ; j++)
			m[i][j] = 0;

	for(i=0; i<4 ; i++)
		m[i][i] = 1;

}

CVMatrix4D::CVMatrix4D(const CVMatrix4D& mat)
{
	for(int i=0; i<4 ; i++)
		for(int j=0; j<4 ; j++)
			m[i][j] = mat.m[i][j];
}

CVMatrix4D::CVMatrix4D(double m00,double m01,double m02,double m03,
					 double m10,double m11,double m12,double m13,
					 double m20,double m21,double m22,double m23, 
					 double m30,double m31,double m32,double m33)
{
	m[0][0] = m00; m[0][1]=m01; m[0][2]=m02;  m[0][3]=m03;
	m[1][0] = m10; m[1][1]=m11; m[1][2]=m12; m[1][3]=m13;
	m[2][0] = m20; m[2][1]=m21; m[2][2]=m22; m[2][3]=m23;
	m[3][0] = m30; m[3][1]=m31; m[3][2]=m32; m[3][3]=m33;
}

CVMatrix4D::CVMatrix4D(const double mat[4][4])
{
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			m[i][j] = mat[i][j];
}

CVMatrix4D::CVMatrix4D(const double mat[16])
{
	for(int i=0; i<4; i++)
		for(int j=0; j<4; j++)
			m[i][j] = mat[i+4*j];
}

CVMatrix4D::CVMatrix4D(CVVect4D vtX, CVVect4D vtY, CVVect4D vtZ)
{
	for (int i=0; i<4; i++) {
		m[0][i] = vtX[i];
		m[1][i] = vtY[i];
		m[2][i] = vtZ[i];
		m[3][i] = 0;
	}
	m[3][3] = 1;

}

CVMatrix4D::~CVMatrix4D(void)
{
}

double CVMatrix4D::m4cof(int r, int c, int r0, int r1, int r2, int c0, int c1, int c2)
{
	return (((r + c) % 2)? -1.0 : 1.0) * M3DET(m[r0][c0], m[r0][c1], m[r0][c2], m[r1][c0], m[r1][c1], m[r1][c2], m[r2][c0], m[r2][c1], m[r2][c2]);
	//return (((r + c) % 2)? -1.0 : 1.0) * M3DET(m[r0][c0], m[r1][c0], m[r2][c0], m[r0][c1], m[r1][c1], m[r2][c1], m[r0][c2], m[r1][c2], m[r2][c2]);
}

double CVMatrix4D::m4det()
{
	double d;

	
	d =	(m[0][0] * m4cof(0,0, 1,2,3, 1,2,3) +
		 m[0][1] * m4cof(0,1, 1,2,3, 0,2,3) +
		 m[0][2] * m4cof(0,2, 1,2,3, 0,1,3) +
		 m[0][3] * m4cof(0,3, 1,2,3, 0,1,2));
	

	//ippmDet_m_64f_4x4((Ipp64f*)m, 32, (Ipp64f*)&d);

	return d;
}

CVMatrix4D CVMatrix4D::Inverse()
{
	CVMatrix4D resM;

	
	double d = m4det();

	if(d == 0.0)
		d = 0.00000000001;
	
	resM.m[0][0] = m4cof(0,0, 1,2,3, 1,2,3) / d;
	resM.m[1][0] = m4cof(0,1, 1,2,3, 0,2,3) / d;
	resM.m[2][0] = m4cof(0,2, 1,2,3, 0,1,3) / d;
	resM.m[3][0] = m4cof(0,3, 1,2,3, 0,1,2) / d;
	
	resM.m[0][1] = m4cof(1,0, 0,2,3, 1,2,3) / d;
	resM.m[1][1] = m4cof(1,1, 0,2,3, 0,2,3) / d;
	resM.m[2][1] = m4cof(1,2, 0,2,3, 0,1,3) / d;
	resM.m[3][1] = m4cof(1,3, 0,2,3, 0,1,2) / d;
	
	resM.m[0][2] = m4cof(2,0, 0,1,3, 1,2,3) / d;
	resM.m[1][2] = m4cof(2,1, 0,1,3, 0,2,3) / d;
	resM.m[2][2] = m4cof(2,2, 0,1,3, 0,1,3) / d;
	resM.m[3][2] = m4cof(2,3, 0,1,3, 0,1,2) / d;

	resM.m[0][3] = m4cof(3,0, 0,1,2, 1,2,3) / d;
	resM.m[1][3] = m4cof(3,1, 0,1,2, 0,2,3) / d;
	resM.m[2][3] = m4cof(3,2, 0,1,2, 0,1,3) / d;
	resM.m[3][3] = m4cof(3,3, 0,1,2, 0,1,2) / d;
	

//	IppStatus is = ippmInvert_m_64f_4x4((Ipp64f*)m, 32, (Ipp64f*)resM.m, 32);

	return resM;
}

double CVMatrix4D::m3cof(int r, int c, int r0, int r1, int c0, int c1) 
{
  return ( ((r + c) % 2)? -1.0 : 1.0) * M2DET(m[r0][c0], m[r0][c1], m[r1][c0], m[r1][c1]);
}

double CVMatrix4D::m3det()
{
	return (M3DET(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2], m[2][0], m[2][1], m[2][2]));
}

CVMatrix4D CVMatrix4D::Inverse3D()
{
	CVMatrix4D resM;

	/*
	double d = m3det();

	resM.m[0][0] = m3cof(0,0, 1,2, 1,2) / d;
	resM.m[1][0] = m3cof(0,1, 1,2, 0,2) / d;
	resM.m[2][0] = m3cof(0,2, 1,2, 0,1) / d;
	
	resM.m[0][1] = m3cof(1,0, 0,2, 1,2) / d;
	resM.m[1][1] = m3cof(1,1, 0,2, 0,2) / d;
	resM.m[2][1] = m3cof(1,2, 0,2, 0,1) / d;
	
	resM.m[0][2] = m3cof(2,0, 0,1, 1,2) / d;
	resM.m[1][2] = m3cof(2,1, 0,1, 0,2) / d;
	resM.m[2][2] = m3cof(2,2, 0,1, 0,1) / d;
	*/

	ippmInvert_m_64f_3x3((Ipp64f*)m, 32, (Ipp64f*)resM.m, 32);

	return resM;
}

CVMatrix4D CVMatrix4D::operator=(const CVMatrix4D& mat)
{
	for(int i=0; i<4 ; i++)
		for(int j=0; j<4 ; j++)
			m[i][j] = mat.m[i][j];

	return (*this);
}

CVMatrix4D CVMatrix4D::operator*(const CVMatrix4D& mat)
{
	CVMatrix4D resM(0,0,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0);

	/*
	for(int i=0; i< 4; i++)
		for(int j=0; j< 4; j++) 
			for(int k=0; k< 4;k++) 
				resM.m[i][j] += m[i][k]*mat.m[k][j];
	*/
	ippmMul_mm_64f_4x4((Ipp64f*)m, 32, (Ipp64f*)mat.m, 32, (Ipp64f*)resM.m, 32);
	
	return resM;
}

BOOL CVMatrix4D::operator==(const CVMatrix4D& mat)
{
	int iRetval = memcmp(m, mat.m, sizeof(m));
	return (iRetval == 0);
}

void CVMatrix4D::SetMatrix4D(double m00,double m01,double m02,double m03,
							double m10,double m11,double m12,double m13,
							double m20,double m21,double m22,double m23, 
							double m30,double m31,double m32,double m33)
{
	m[0][0] = m00; m[0][1]=m01; m[0][2]=m02; m[0][3]=m03;
	m[1][0] = m10; m[1][1]=m11; m[1][2]=m12; m[1][3]=m13;
	m[2][0] = m20; m[2][1]=m21; m[2][2]=m22; m[2][3]=m23;
	m[3][0] = m30; m[3][1]=m31; m[3][2]=m32; m[3][3]=m33;
}

void CVMatrix4D::LoadIdentity(void)
{
	for(int i=0; i<4 ; i++)
		for(int j=0; j<4 ; j++)
			m[i][j] = 0;

	for(i=0; i<4 ; i++)
		m[i][i] = 1;
}

void CVMatrix4D::Scale(double sx, double sy, double sz)
{
	CVMatrix4D tempM(sx,0,0,0, 0,sy,0,0, 0,0,sz,0, 0,0,0,1);
	*this = tempM*(*this);
}

void CVMatrix4D::Translate(double tx, double ty, double tz)
{
	CVMatrix4D tempM(1,0,0,tx, 0,1,0,ty, 0,0,1,tz, 0,0,0,1);
	*this = tempM*(*this);
}

// angle is degree
void CVMatrix4D::Rotate(int axis, double degree)
{
	CVMatrix4D tempM;
	double angle=(degree/180.)*M_PI;

	switch(axis){
	case X_AXIS:
		tempM.m[1][1] = cos(angle);  tempM.m[2][1] = sin(angle);
		tempM.m[1][2] = -sin(angle); tempM.m[2][2] = cos(angle);
		break;

	case Y_AXIS:
		tempM.m[0][0] = cos(angle); tempM.m[2][0] = -sin(angle);
		tempM.m[0][2] = sin(angle); tempM.m[2][2] = cos(angle);
		break;

	case Z_AXIS:
		tempM.m[0][0] = cos(angle); tempM.m[1][0] = sin(angle);
		tempM.m[0][1] = -sin(angle);tempM.m[1][1] = cos(angle);
		break;
	}
	
	*this = tempM*(*this);
}

void CVMatrix4D::Rotate(CVVect4D axis, double degree)
{
	CVMatrix4D mxQuaternion;

	degree *= 0.5;

	axis.Normalize();

	double x = axis.m[0] * sin(degree);
	double y = axis.m[1] * sin(degree);
	double z = axis.m[2] * sin(degree);
	double w = cos(degree);

	mxQuaternion.SetMatrix4D(	w*w + x*x - y*y - z*z,	2*x*y - 2*w*z,			2*x*z + 2*w*y,			0,
								2*x*y + 2*w*z,			w*w - x*x + y*y - z*z,	2*y*z - 2*w*x,			0,
								2*x*z - 2*w*y,			2*y*z + 2*w*x,			w*w - x*x - y*y + z*z,	0,
								0,						0,						0,						w*w + x*x + y*y + z*z);

	*this = mxQuaternion * (*this);
}


double CVMatrix4D::CalcuSx()
{
	return (m[1][1]*m[0][2] - m[0][1]*m[1][2])/(m[0][0]*m[1][1]-m[0][1]*m[1][0]);
}

double CVMatrix4D::CalcuSy()
{
	return (m[0][0]*m[1][2] - m[1][0]*m[0][2])/(m[0][0]*m[1][1] - m[0][1]*m[1][0]);
}

CVMatrix4D CVMatrix4D::Warp2D(double ti, double tj)
{
	CVMatrix4D resM;

	resM.SetMatrix4D(m[0][0], m[0][1], m[0][3]-ti*m[0][0]-tj*m[0][1], 0,
					 m[1][0], m[1][1], m[1][3]-ti*m[1][0]-tj*m[1][1], 0,
					  0,0,1,0,
					 0,0,0,1);

	return resM;
}

double CVMatrix4D::ViewVectX()
{
	return m[0][1] * m[1][2] - m[1][1] * m[0][2];
}

double CVMatrix4D::ViewVectY()
{
	return m[1][0] * m[0][2] - m[0][0] * m[1][2];
}

double CVMatrix4D::ViewVectZ()
{
	return m[0][0] * m[1][1] - m[1][0] * m[0][1];
}

void CVMatrix4D::TransformVect3D(double x, double y,double z, double *px, double *py, double *pz)
{
	*px = x * m[0][0] + y * m[0][1] + z * m[0][2];
	*py = x * m[1][0] + y * m[1][1] + z * m[1][2];
	*pz = x * m[2][0] + y * m[2][1] + z * m[2][2];
}

CVVect4D CVMatrix4D::TransformVect3D(const CVVect4D vect)
{
	CVVect4D resV(0,0,0);

	resV.m[0] = vect[0] * m[0][0] + vect[1] * m[0][1] + vect[2] * m[0][2];
	resV.m[1] = vect[0] * m[1][0] + vect[1] * m[1][1] + vect[2] * m[1][2];
	resV.m[2] = vect[0] * m[2][0] + vect[1] * m[2][1] + vect[2] * m[2][2];

	return resV;
}

CVVect4D CVMatrix4D::operator*(const CVVect4D& vect)
{
	CVVect4D resV(0,0,0,0);

	/*
	for(int i=0; i< 4; i++)
		for(int j=0; j< 4; j++) 
			resV.m[i] += m[i][j]*vect[j];
	*/
	ippmMul_mv_64f_4x4((Ipp64f*)m, 32, (Ipp64f*)vect.m, (Ipp64f*)resV.m);
	
	return resV;
}

CVPoint3D <double> CVMatrix4D::operator*(const CVPoint3D <double> pt3d)
{
	CVVect4D vect(pt3d.x, pt3d.y, pt3d.z);
	CVVect4D resV(0,0,0,0);

	/*for(int i=0; i< 4; i++)
		for(int j=0; j< 4; j++) 
			resV.m[i] += m[i][j]*vect[j];*/

	ippmMul_mv_64f_4x4((Ipp64f*)m, 32, (Ipp64f*)vect.m, (Ipp64f*)resV.m);
	
	return CVPoint3D <double>(resV);
}

int CVMatrix4D::GetPrincipalAxis(double fRatio)
{
	double v_x, v_y, v_z;

	v_x = m[0][1] * m[1][2] - m[1][1] * m[0][2];
	v_y = m[1][0] * m[0][2] - m[0][0] * m[1][2];
	v_z = m[0][0] * m[1][1] - m[1][0] * m[0][1];
	
	v_x = fabs(v_x);
	v_y = fabs(v_y);
	v_z = fabs(v_z)*fRatio;
	
	if ( v_z >= v_y ) {
		if ( v_z >= v_x ) return Z_AXIS;
		else return X_AXIS;
	} else {
		if ( v_y >= v_x ) return Y_AXIS;
		else return X_AXIS;
	}
}

int CVMatrix4D::GetPrincipalAxisWithSign(double fRatio)
{
	double v_x, v_y, v_z;

	v_x = m[0][1] * m[1][2] - m[1][1] * m[0][2];
	v_y = m[1][0] * m[0][2] - m[0][0] * m[1][2];
	v_z = m[0][0] * m[1][1] - m[1][0] * m[0][1];
	
//	v_x = fabs(v_x);
//	v_y = fabs(v_y);
//	v_z = fabs(v_z)*fRatio;

	// x : 1	y : 2	z : 3
	
	if ( fabs(v_z) >= fabs(v_y) ) {
		if ( fabs(v_z) >= fabs(v_x) ) {
			if (v_z >= 0)	return 3;
			else			return -3;
		}
		else {
			if (v_x >= 0)	return 1;
			else			return -1;
		}
	} else {
		if ( fabs(v_y) >= fabs(v_x) ) {
			if (v_y >= 0)	return 2;
			else			return -2;
		}
		else {
			if (v_x >= 0)	return 1;
			else			return -1;
		}
	}
}

CVMatrix4D CVMatrix4D::Transpose()
{
	CVMatrix4D resM;
	for(int i=0; i<4; i++) {
		resM.m[0][i] = m[i][0];
		resM.m[1][i] = m[i][1];
		resM.m[2][i] = m[i][2];
		resM.m[3][i] = m[i][3];
	}
	return resM;
}

CVMatrix4D CVMatrix4D::GetRotateSubMatrix()
{
	CVMatrix4D resM;

	CVVect4D scale(1,0,0,0);
	for(int i=0; i<3; i++) {
		resM.m[0][i] = m[0][i];
		resM.m[1][i] = m[1][i];
		resM.m[2][i] = m[2][i];
	}

	return resM;
}

void CVMatrix4D::SetTranslateSubMatrix(CVMatrix4D mxTranslateSub)
{
	CVMatrix4D resM;

	for(int i=0; i<3; i++) 
		m[i][3] = mxTranslateSub.m[i][3];
}
/*
CVVect4D CVMatrix4D::GetScaleFactor()
{
	CVVect4D vec(1,1,1);

	return vec;
}
*/
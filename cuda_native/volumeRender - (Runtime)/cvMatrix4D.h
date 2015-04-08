#ifndef __CI_MATRIX4D_H__
#define __CI_MATRIX4D_H__

#include "cvVect4D.h"
#include "cvPrimitive.h"

#define X_AXIS	0
#define Y_AXIS	1
#define Z_AXIS	2
#define M_PI 3.1415926535897932384626433832795

class CLASS_DECL_API CVMatrix4D
{
public:
	double m[4][4];

private:
	double m4cof(int r, int c, int r0, int r1, int r2, int c0, int c1, int c2);
	double m3cof(int r, int c, int r0, int r1, int c0, int c1); 
	double m3det(void);	
	double m4det(void);

public:
	CVMatrix4D();
	CVMatrix4D(const CVMatrix4D& mat);
	CVMatrix4D(const double mat[4][4]);
	CVMatrix4D(const double mat[16]);
	CVMatrix4D(double m00,double m01,double m02,double m03,double m10,double m11,double m12,double m13,double m20,double m21,double m22,double m23, double m30,double m31,double m32,double m33);
	CVMatrix4D(CVVect4D vtX, CVVect4D vtY, CVVect4D vtZ);
	~CVMatrix4D();

	CVMatrix4D operator=(const CVMatrix4D& mat);
	CVMatrix4D operator*(const CVMatrix4D& mat);
	CVVect4D operator*(const CVVect4D& vect);
	CVPoint3D <double> operator*(const CVPoint3D <double> pt3d);
	BOOL operator==(const CVMatrix4D& mat);
	
	CVMatrix4D Inverse(void);
	CVMatrix4D Inverse3D(void);

	double GetDet(void) {
		return m4det();
	}

	void LoadIdentity(void);
	void Scale(double sx, double sy, double sz);
	void Translate(double tx, double ty, double tz);
	void Rotate(int axis, double degree);
	void Rotate(CVVect4D axis, double degree);

	void SetMatrix4D(double m00,double m01,double m02,double m03,double m10,double m11,double m12,double m13,double m20,double m21,double m22,double m23, double m30,double m31,double m32,double m33);
	
	double CalcuSx();
	double CalcuSy();
	
	CVMatrix4D Warp2D(double ti, double tj);
	
	double ViewVectX();
	double ViewVectY();
	double ViewVectZ();
	
	CVMatrix4D Transpose();
	void TransformVect3D(double x, double y,double z, double *pa, double *py, double *pz);
	CVVect4D	TransformVect3D(const CVVect4D vect);

	int GetPrincipalAxis(double fRatioZ=1.);
	int GetPrincipalAxisWithSign(double fRatio=1.);
	CVMatrix4D GetRotateSubMatrix();
	void SetTranslateSubMatrix(CVMatrix4D mxTranslateSub);

	CVVect4D	GetVectX() {return CVVect4D(m[0][0], m[0][1], m[0][2], m[0][3]);}
	CVVect4D	GetVectY() {return CVVect4D(m[1][0], m[1][1], m[1][2], m[1][3]);}
	CVVect4D	GetVectZ() {return CVVect4D(m[2][0], m[2][1], m[2][2], m[2][3]);}

//	CVVect4D	GetScaleFactor();

};
#endif //__CI_MATRIX4D_H__
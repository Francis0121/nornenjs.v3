// cvVect4D.h: interface for the CVVect4D class.
//
//////////////////////////////////////////////////////////////////////

#ifndef __CV_VECT4D_H__
#define __CV_VECT4D_H__

class CLASS_DECL_API CVVect4D {

public:
	int GetSecondaryAxisWithSign(int iPrincipal);
	int GetPrincipalAxisWithSign();
	int GetLeastPrincipalAxis();
	int GetPrincipalAxis();
	double m[4];

	CVVect4D();
	CVVect4D(const CVVect4D& v);
	CVVect4D(double m0,double m1,double m2, double m3);
	CVVect4D(double m0,double m1,double m2);
	CVVect4D(double vec[4]);
	CVVect4D(CPoint pt);
	~CVVect4D();

	CVVect4D operator=(const CVVect4D& vect);

	CVVect4D operator+(const CVVect4D& vect);
	CVVect4D operator-(const CVVect4D& vect);

	double operator*(const CVVect4D& vect);
	CVVect4D operator*(const double& var);

	// sungeui start
	CVVect4D operator/(const double& var);
	double Magnitude (void);
	CVVect4D CrossProduct (const CVVect4D& vect);
	double DotProduct (const CVVect4D& vect);
	// sungeui end

	//double What(int inx);
	double operator[](const int& inx) const;
	BOOL operator==(const CVVect4D &r) const;
	BOOL operator!=(const CVVect4D &r) const;
	BOOL IsParallel(CVVect4D r);

	void Normalize(void);
	void ToCartesianCoord(void);

	CVVect4D Inverse();
protected:
};
#endif //__CV_VECT4D_H__
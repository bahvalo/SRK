#include "asrk.h"
using namespace std;


// Check some order conditions for a Butcher table
void tIMEXMethod::OrderCheck(int NumStages, const vector<double>& A){
    if(int(A.size()) != NumStages*(NumStages+1)) { printf("OrderCheck: wrong input\n"); exit(0); }
    const double* b = A.data() + NumStages*NumStages;
    double sum1 = 0., sum2 = 0., sum31 = 0., sum32 = 0.;

    // First order condition
    for(int k=0; k<NumStages; k++) sum1 += b[k];
    sum1 -= 1.0;

    // Second order condition
    for(int j=0; j<NumStages; j++) for(int k=0; k<NumStages; k++) sum2 += 2.*b[j]*A[j*NumStages+k];
    sum2 -= 1.0;

    // Third order conditions (H.,N.,W., Section II.2)
    for(int j=0; j<NumStages; j++) for(int k=0; k<NumStages; k++) for(int l=0; l<NumStages; l++)
        sum31 += 3.*b[j]*A[j*NumStages+k]*A[j*NumStages+l];
    sum31 -= 1.0;

    for(int j=0; j<NumStages; j++) for(int k=0; k<NumStages; k++) for(int l=0; l<NumStages; l++)
        sum32 += 6.*b[j]*A[j*NumStages+k]*A[k*NumStages+l];
    sum32 -= 1.0;

    printf("sums: %.2e, %.2e, %.2e, %.2e\n", sum1, sum2, sum31, sum32);
}


// Calculate coefficients `d`, which are used for methods of type CK
vector<double> tIMEXMethod::CalcD(int NumStages, const vector<double>& A){
    vector<double> d(NumStages);
    double a_ss = A[NumStages*NumStages-1];
    d[0] = 1.;
    for(int j=1; j<NumStages; j++){
        d[j] = 0.;
        for(int k=0; k<j; k++) d[j] -= A[j*NumStages+k]*d[k]/a_ss;
    }
    return d;
}


// Ascher, Ruuth, Spiteri. IMEX RK method for time-dependent PDEs. 1997
// Implicit-explicit Euler method
ARS_121::ARS_121(){
    NStages = 2;
    IMEXtype = tIMEXtype::IMEX_ARS;
    ButcherE = vector<double>({
        0., 0.,
        1., 0.,
        0., 1.
    });
    ButcherI = vector<double>({
        0., 0.,
        0., 1.,
        0., 1.
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// ARS (2,3,2) method
ARS_232::ARS_232(){
    NStages = 3;
    IMEXtype = tIMEXtype::IMEX_ARS;
    const double gamma = 1.-0.5*sqrt(2.);
    const double delta = -2.*sqrt(2.)/3.;
    ButcherE = vector<double>({
           0.,       0.,    0.,
        gamma,       0.,    0.,
        delta, 1.-delta,    0.,
           0., 1.-gamma, gamma
    });
    ButcherI = vector<double>({
           0.,       0.,    0.,
           0.,    gamma,    0.,
           0., 1.-gamma, gamma,
           0., 1.-gamma, gamma
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// ARS (3,4,3) method
ARS_343::ARS_343(){
    NStages = 4;
    IMEXtype = tIMEXtype::IMEX_ARS;
    const double gamma = 0.435866521508458999416019451193556842529; // middle root of 6x^3-18x^2+9x-1=0
    const double b1 = -1.5*gamma*gamma + 4*gamma - 0.25;
    const double b2 = 1.5*gamma*gamma - 5.*gamma + 1.25;
    ButcherI = vector<double>({
           0.,            0.,    0.,    0.,
           0.,         gamma,    0.,    0.,
           0., (1.-gamma)/2., gamma,    0.,
           0.,            b1,    b2, gamma,
           0.,            b1,    b2, gamma
    });
    const double a42 = 0.5529291479;
    const double a43 = a42;
    const double a31 = (1.-4.5*gamma+1.5*gamma*gamma)*a42 + (2.75-10.5*gamma+3.75*gamma*gamma)*a43 - 3.5+13.*gamma-4.5*gamma*gamma;
    const double a32 = -(1.-4.5*gamma+1.5*gamma*gamma)*a42 - (2.75-10.5*gamma+3.75*gamma*gamma)*a43 + 4.-12.5*gamma+4.5*gamma*gamma;
    ButcherE = vector<double>({
              0.,        0.,    0.,    0.,
           gamma,        0.,    0.,    0.,
             a31,       a32,    0.,    0.,
      1.-a42-a43,       a42,   a43,    0.,
              0.,        b1,    b2, gamma
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// Ascher, Ruuth, Spiteri. IMEX RK method for time-dependent PDEs. 1997
// Methods that satisfy \hat{b} = \hat{a}_{s,*} instead of b = \hat{b}
// ARS (1,1,1) method
ARS_111::ARS_111(){
    NStages = 2;
    IMEXtype = tIMEXtype::IMEX_ARS;
    ButcherE = vector<double>({
        0., 0.,
        1., 0.,
        1., 0.
    });
    ButcherI = vector<double>({
        0., 0.,
        0., 1.,
        0., 1.
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 1.;
}

// ARS (2,2,2) method
ARS_222::ARS_222(){
    NStages = 3;
    IMEXtype = tIMEXtype::IMEX_ARS;
    const double gamma = 1.-0.5*sqrt(2.);
    const double delta = 1.-1./(2.*gamma);
    ButcherE = vector<double>({
           0.,       0.,    0.,
        gamma,       0.,    0.,
        delta, 1.-delta,    0.,
        delta, 1.-delta,    0.
    });
    ButcherI = vector<double>({
           0.,       0.,    0.,
           0.,    gamma,    0.,
           0., 1.-gamma, gamma,
           0., 1.-gamma, gamma
    });
    alpha_tau_max[0] = 0.82;
    alpha_tau_max[1] = 0.0;
}

// ARS (4,4,3) method
ARS_443::ARS_443(){
    NStages = 5;
    IMEXtype = tIMEXtype::IMEX_ARS;
    ButcherE = vector<double>({
             0.,      0.,      0.,      0.,      0.,
          1./2.,      0.,      0.,      0.,      0.,
        11./18.,  1./18.,      0.,      0.,      0.,
          5./6.,  -5./6.,   1./2.,      0.,      0.,
          1./4.,   7./4.,   3./4.,  -7./4.,      0.,
          1./4.,   7./4.,   3./4.,  -7./4.,      0.
    });
    ButcherI = vector<double>({
             0.,      0.,      0.,      0.,      0.,
             0.,   1./2.,      0.,      0.,      0.,
             0.,   1./6.,   1./2.,      0.,      0.,
             0.,  -1./2.,   1./2.,   1./2.,      0.,
             0.,   3./2.,  -3./2.,   1./2.,   1./2.,
             0.,   3./2.,  -3./2.,   1./2.,   1./2.
    });
    alpha_tau_max[0] = 2.;
    alpha_tau_max[1] = 1.43;
}


// Boscarino, Russo. On the uniform accuracy of IMEX RK schemes and applications to hyperbolic systems with relaxation. 2007
// MARS (=modified ARS) (3,4,3) method
MARS_343::MARS_343(){
    NStages = 4;
    IMEXtype = tIMEXtype::IMEX_ARS;
    const double gamma = 0.435866521508458999416019451193556842529; // middle root of 6x^3-18x^2+9x-1=0
    const double b2 = 1.20849664917601;
    ButcherI = vector<double>({
        0.0,                0.0,               0.0,            0.0,
        0.0,              gamma,               0.0,            0.0,
        0.0,   0.28206673924577,             gamma,            0.0,
        0.0,                 b2,       1.-gamma-b2,          gamma,
        0.0,                 b2,       1.-gamma-b2,          gamma
    });
    ButcherE = vector<double>({
                     0.0,                0.0,               0.0,            0.0,
                   gamma,                0.0,               0.0,            0.0,
        0.535396540307354, 0.182536720446875,               0.0,            0.0,
        0.63041255815287, -0.83193390106308,   1.20152134291021,            0.0,
        0.0,                 b2,       1.-gamma-b2,          gamma
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// ARK3(2)4L[2]SA scheme
ARK3::ARK3(){
    NStages = 4;
    IMEXtype = tIMEXtype::IMEX_CK;
    const double gamma = 0.435866521508458999416019451193556842529; // middle root of 6x^3-18x^2+9x-1=0
    const double b1 = 1471266399579./7840856788654.;
    const double b2 = -4482444167858./7529755066697.;
    const double b3 = 1.-gamma-b1-b2;
    ButcherI = vector<double>({
                      0.0,                0.0,               0.0,            0.0,
                    gamma,              gamma,               0.0,            0.0,
        2746238789719./10658868560708.,  -640167445237./6845629431997.,  gamma, 0.0,
        b1, b2, b3, gamma,
        b1, b2, b3, gamma
    });
    ButcherE = vector<double>({
                      0.0,                0.0,               0.0,            0.0,
                  2*gamma,                0.0,               0.0,            0.0,
        5535828885825./10492691773637., 788022342437./10882634858940., 0.0, 0.0,
        6485989280629./16251701735622., -4246266847089./9704473918619., 10755448449292./10357097424841., 0.0,
        b1, b2, b3, gamma
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// ARK4(3)6L[2]SA scheme
ARK4::ARK4(){
    NStages = 6;
    IMEXtype = tIMEXtype::IMEX_CK;
    const double b[6] = {82889./524892., 0., 15625./83664., 69875./102672., -2260./8211., 0.25};
    ButcherI = vector<double>({
                      0.0,                0.0,               0.0,            0.0,            0.0,            0.0,
                     0.25,               0.25,               0.0,            0.0,            0.0,            0.0,
             8611./62500.,      -1743./31250.,              0.25,            0.0,            0.0,            0.0,
       5012029./34652500.,  -654441./2922500.,   174375./388108.,           0.25,            0.0,            0.0,
       15267082809./155376265600., -71443401./120774400., 730878875./902184768., 2285395./8070912., 0.25,    0.0,
                     b[0],               b[1],               b[2],           b[3],          b[4],           b[5],
                     b[0],               b[1],               b[2],           b[3],          b[4],           b[5]
    });
    ButcherE = vector<double>({
                      0.0,                0.0,               0.0,            0.0,            0.0,            0.0,
                      0.5,                0.0,               0.0,            0.0,            0.0,            0.0,
            13861./62500.,       6889./62500.,               0.0,            0.0,            0.0,            0.0,
            -116923316275./2393684061468., -2731218467317./15368042101831., 9408046702089./11113171139209., 0.0, 0.0, 0.0,
            -451086348788./2902428689909., -2682348792572./7519795681897., 12662868775082./11960479115383., 3355817975965./11060851509271., 0.0, 0.0,
            647845179188./3216320057751., 73281519250./8382639484533., 552539513391./3454668386233., 3354512671639./8306763924573., 4040./17871., 0.0,
                     b[0],               b[1],               b[2],           b[3],          b[4],           b[5]
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// ARK5(4)8L[2]SA scheme
ARK5::ARK5(){
    NStages = 8;
    IMEXtype = tIMEXtype::IMEX_CK;
    const double gamma = 41./200.;
    const double b[8] = {-872700587467./9133579230613., 0., 0., 22348218063261./9555858737531., -1143369518992./8141816002931., -39379526789629./19018526304540., 32727382324388./42900044865799., gamma};
    ButcherI = vector<double>({
        0., 0., 0., 0., 0., 0., 0., 0.,
        gamma, gamma, 0., 0., 0., 0., 0., 0.,
        41./400., -567603406766./11931857230679., gamma, 0., 0., 0., 0., 0.,
        683785636431./9252920307686., 0., -110385047103./1367015193373., gamma, 0., 0., 0., 0.,
        3016520224154./10081342136671., 0., 30586259806659./12414158314087., -22760509404356./11113319521817., gamma, 0., 0., 0.,
        218866479029./1489978393911., 0., 638256894668./5436446318841., -1179710474555./5321154724896., -60928119172./8023461067671., gamma, 0., 0.,
        1020004230633./5715676835656., 0., 25762820946817./25263940353407., -2161375909145./9755907335909., -211217309593./5846859502534., -4269925059573./7827059040749., gamma, 0.,
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
    });
    ButcherE = vector<double>({
        0., 0., 0., 0., 0., 0., 0., 0.,
        2.*gamma, 0., 0., 0., 0., 0., 0., 0.,
        367902744464./2072280473677., 677623207551./8224143866563., 0., 0., 0., 0., 0., 0.,
        1268023523408./10340822734521., 0., 1029933939417./13636558850479., 0., 0., 0., 0., 0.,
        14463281900351./6315353703477., 0., 66114435211212./5879490589093., -54053170152839./4284798021562., 0., 0., 0., 0.,
        14090043504691./34967701212078., 0., 15191511035443./11219624916014., -18461159152457./12425892160975., -281667163811./9011619295870., 0., 0., 0.,
        19230459214898./13134317526959., 0., 21275331358303./2942455364971., -38145345988419./4862620318723., -0.125, -0.125, 0., 0.,
        -19977161125411./11928030595625., 0., -40795976796054./6384907823539., 177454434618887./12078138498510., 782672205425./8267701900261., -69563011059811./9646580694205., 7356628210526./4942186776405., 0.,
        b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// MARK3(2)4L[2]SA scheme
MARK3::MARK3(){
    NStages = 4;
    IMEXtype = tIMEXtype::IMEX_CK;
    const double gamma = 0.435866521508458999416019451193556842529; // middle root of 6x^3-18x^2+9x-1=0
    const double b1 = 0.60424832458800;
    ButcherI = vector<double>({
                      0.0,                0.0,               0.0,            0.0,
                    gamma,              gamma,               0.0,            0.0,
        -4.30002662176923,   2.26541338346372,             gamma,            0.0,
                       b1,                0.0,       1.-gamma-b1,          gamma,
                       b1,                0.0,       1.-gamma-b1,          gamma
    });
    ButcherE = vector<double>({
                      0.0,                0.0,               0.0,            0.0,
                  2*gamma,                0.0,               0.0,            0.0,
        -3.06478674186224,   1.46604002506519,               0.0,            0.0,
         0.21444560762133,   0.71075364965269,  0.07480074272597,            0.0,
                       b1,                0.0,       1.-gamma-b1,          gamma
    });
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// Boscarino. On an accurate third order IMEX RK method for stiff problems. 2009
// Boscarino (5,5,3) method
BHR_553::BHR_553(){
    NStages = 5;
    ButcherI.resize(30);
    ButcherE.resize(30);
    IMEXtype = tIMEXtype::IMEX_CK;
    const double gamma = 424782./974569.;
    ButcherI[5]=ButcherI[6]=gamma;
    ButcherI[10]=gamma; ButcherI[11]=-31733082319927313./455705377221960889379854647102.; ButcherI[12]=gamma;
    ButcherI[15]=-3012378541084922027361996761794919360516301377809610./45123394056585269977907753045030512597955897345819349.;
    ButcherI[16]=-62865589297807153294268./102559673441610672305587327019095047.;
    ButcherI[17]=418769796920855299603146267001414900945214277000./212454360385257708555954598099874818603217167139.;
    ButcherI[18]=gamma;
    ButcherI[20]=487698502336740678603511./1181159636928185920260208.;
    ButcherI[22]=302987763081184622639300143137943089./1535359944203293318639180129368156500.;
    ButcherI[23]=-105235928335100616072938218863./2282554452064661756575727198000.;
    ButcherI[24]=gamma;
    ButcherE[5]=2.*gamma;
    ButcherE[10]=gamma; ButcherE[11]=gamma;
    ButcherE[15]=-475883375220285986033264./594112726933437845704163.;
    ButcherE[17]=1866233449822026827708736./594112726933437845704163.;
    ButcherE[20]=62828845818073169585635881686091391737610308247./176112910684412105319781630311686343715753056000.;
    ButcherE[21]=-ButcherI[22];
    ButcherE[22]=262315887293043739337088563996093207./297427554730376353252081786906492000.;
    ButcherE[23]=-987618231894176581438124717087./23877337660202969319526901856000.;
    for(int i=0; i<5; i++) ButcherE[25+i]=ButcherI[25+i]=ButcherI[20+i];
    alpha_tau_max[0] = alpha_tau_max[1] = 2.;
}

// Pareschi, Russo. High order asymptotically strong-stability-preserving methods for hyperbolic systems with stiff relaxation. 2003
SSP2_322::SSP2_322(){
    NStages = 3;
    IMEXtype = tIMEXtype::IMEX_A;
    ButcherE = vector<double>({
         0.0,  0.0,  0.0,
         0.0,  0.0,  0.0,
         0.0,  1.0,  0.0,
         0.0,  0.5,  0.5});
    ButcherI = vector<double>({
         0.5,  0.0,  0.0,
        -0.5,  0.5,  0.0,
         0.0,  0.5,  0.5,
         0.0,  0.5,  0.5});
    alpha_tau_max[0] = 0.66;
    alpha_tau_max[1] = 0.;
}

// Boscarino Qiu Russo Xiong. High Order Semi-implicit WENO Schemes for All Mach Full Euler System of Gas Dynamics. 2021
// SI-IMEX (3,3,2) method
// The same tables as for ARS_232, except a_{0,0} = gamma (and therefore the method is of type A)
SI_IMEX_332::SI_IMEX_332(){
    NStages = 3;
    IMEXtype = tIMEXtype::IMEX_A;
    const double gamma = 1.-0.5*sqrt(2.);
    const double delta = -2.*sqrt(2.)/3.;
    ButcherE = vector<double>({
           0.,       0.,    0.,
        gamma,       0.,    0.,
        delta, 1.-delta,    0.,
           0., 1.-gamma, gamma
    });
    ButcherI = vector<double>({
        gamma,       0.,    0.,
           0.,    gamma,    0.,
           0., 1.-gamma, gamma,
           0., 1.-gamma, gamma
    });
    alpha_tau_max[0] = 0.82;
    alpha_tau_max[1] = 0.0;
}

// SI-IMEX (4,4,3) method
// The implicit table is the same as for ARS_343, except a_{0,0} = gamma (and therefore the method is of type A).
// Looks like the explicit table is in error, since the third order conditions are not met
SI_IMEX_443::SI_IMEX_443(){
    NStages = 4;
    IMEXtype = tIMEXtype::IMEX_A;
    const double gamma = 0.435866521508458999416019451193556842529; // middle root of 6x^3-18x^2+9x-1=0
    const double b2 = 1.20849664917601;
    ButcherI = vector<double>({
      gamma,                0.0,               0.0,            0.0,
        0.0,              gamma,               0.0,            0.0,
        0.0,   0.28206673924577,             gamma,            0.0,
        0.0,                 b2,       1.-gamma-b2,          gamma,
        0.0,                 b2,       1.-gamma-b2,          gamma
    });
    ButcherE = vector<double>({
                     0.0,                0.0,               0.0,            0.0,
                   gamma,                0.0,               0.0,            0.0,
          0.435866521508,     0.282066739245,               0.0,            0.0,
      -0.733534082748750,     2.150527381100,   -0.416993298352,            0.0,
                     0.0,                 b2,       1.-gamma-b2,          gamma
    });
    alpha_tau_max[0] = 1.09;
    alpha_tau_max[1] = 1.08;
}

// Boscarino Cho. Asymptotic Analysis of IMEX-RK Methods for ES-BGK Model at Navier-Stokes level. 2024
// The implicit table is the same as for ARS_343, except a_{0,0} = gamma (and therefore the method is of type A).
SI_IMEX_433::SI_IMEX_433(){
    NStages = 4;
    IMEXtype = tIMEXtype::IMEX_A;
    const double gamma = 0.435866521508458999416019451193556842529; // middle root of 6x^3-18x^2+9x-1=0
    const double b2 = 1.20849664917601;
    ButcherI = vector<double>({
                  gamma,                0.0,               0.0,            0.0,
                    0.0,              gamma,               0.0,            0.0,
                    0.0,   0.28206673924577,             gamma,            0.0,
                    0.0,                 b2,       1.-gamma-b2,          gamma,
                    0.0,                 b2,       1.-gamma-b2,          gamma
    });
    ButcherE = vector<double>({
                     0.0,                0.0,                0.0,          0.0,
                   gamma,                0.0,                0.0,          0.0,
       1.243893189483362, -0.525959928729133,                0.0,          0.0,
       0.630412558152867,  0.786580740199155, -0.416993298352022,          0.0,
                     0.0,                 b2,        1.-gamma-b2,        gamma
    });
    alpha_tau_max[0] = 1.09;
    alpha_tau_max[1] = 1.08;
}

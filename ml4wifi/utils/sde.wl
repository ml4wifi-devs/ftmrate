(* wolframscript  -f sde.wl *)



proc = ItoProcess[{DifferentialD[\[Nu][t]] == \[Sigma]1 DifferentialD[w1[t]]
    , DifferentialD[\[Rho][t]] == \[Nu][t] DifferentialD[t] + \[Sigma]2 DifferentialD[w2[t]]}, 
    {\[Nu][t], \[Rho][t]}, 
    {{\[Nu], \[Rho]}, {\[Nu]0, \[Rho]0}}, 
    t, 
    {w1 \[Distributed] WienerProcess[], w2 \[Distributed] WienerProcess[]}
    ]

Print["Mean\n"]
Print @ ToString @ TeXForm @ MatrixForm @ Mean[proc[\[Tau]]]


Print["Cov:\n"]
Print @ ToString @ TeXForm @ MatrixForm @ Covariance[proc[\[Tau]]] 


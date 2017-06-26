 ("ALPHA [0,6] COMPUTATION")
            SPR = 14.0;  B = Min[ 16.0, 30.0/(1+0.0) - x ];  A = Min[ 12.0, 26.0/(1+0.0) - x, 28.0/(1+0.0) - y, 42.0/(1+0.0) - x - y ];  mu = { { 15.0, 17.0, 13.0 } };  sig = { { 1.0, 0.0, 0.0 }, { 0.0, 4.0, 0.0 }, { 0.0, 0.0, 0.25 } };
            h = 3;  r = { {x, y, z} };  intlimit = Infinity;
            mulres = ((r - mu).Inverse[sig].Transpose[r - mu])[[1,1]];
            f = Exp[-1/2 * mulres]/Sqrt[(2*Pi)^h * Det[sig]];
            intres = Integrate[f, {x, -intlimit, SPR}, {y, -intlimit, B}, {z, -intlimit, A}];
            Print[ToString[AccountingForm[intres, 16]]]; (" OUTPUT ALPHA [0,6]")
            
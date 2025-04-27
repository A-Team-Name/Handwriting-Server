⎕IO←0
⎕PW←12345

'dec' 'disp' 'displays'⎕CY'dfns'
'assign'⎕CY'dfns'

Atan2←12○⊣+0J1×⊢ ⍝ x Atan2 y | https://aplcart.info?q=atan2
_tr←{⍵⊣⎕←⍺⍺ ⍵}   ⍝ logging, eg ⍴_tr
Atan←{ ⍝ x Atan y
	⎕DIV←1
	+∘(○0>⊢)⍨¯3○(○.5)@(⍺=0⍨)⍵÷⍺
}

Load←{
	⍝ ⍺: input shape
	⍝ ⍵: input path
	⍝ ←: bitarray
	tie←⍵ ⎕NTIE 0
	data←0≤(⍺⍴⎕NREAD tie 83 ¯1)
	_←⎕NUNTIE tie
	data
}

Split←{
	⍝ ⍵: bitarray to split
	⍝ ←: list of split bitarrays

	⍝ TODO
	⍝ - [x] iterate over i,j pairs
	⍝ - [ ] do first row and column independently, avoid validity check for later rows

	data←⍵
	(h w)←⍴data
	p←(⍉w h⍴⍳h),⍤0⊢h w⍴⍳w
	_←{
		(i j)←⍵
		~data[i;j]: ⍵
		k←(i≥1)(j≥1)/(¯1 0)(0 ¯1)+¨⊂(i j)
		k/⍨←data[k]
		0=≢k: ⍵
		k←⌷∘p¨⍣≡k
		p[i;j;]←⊃k
		1=≢k: ⍵
		((1⊃k)⌷p)←⊃k
		⍵
	}⍤1⊢p
	p←,+/w 1×⍤1⊢p
	p←{p[⍵]}⍣≡p
	s←(,data)×1+p

	c←{⍵[⍋⍵]}∪s
	s←c⍳s
	m←↓s∘.=⍨1+⍳¯1+≢c
	i←w|⍸¨m
	min←⌊/¨i
	max←⌈/¨i
	_←m{(⍺/s)←⍵}¨1+⌊/⍤⍸⍤1(⊢∨∨.∧⍨)⍣≡∨∘⍉⍨(min∘.≥min)∧(max∘.≤max)∨min∘.≤min+0.5×max-min

	⍝ FIXME: not so much duplicate computation please
	c←{⍵[⍋⍵]}∪s
	s←c⍳s
	m←↓s∘.=⍨1+⍳¯1+≢c
	i←w|⍸¨m
	min←⌊/¨i
	max←⌈/¨i
	g←(0,0,⍨0⍪0⍪⍨⊢)¨(∨⌿⊢⍤/⊢)¨h w∘⍴¨m
	g←g[⍋min] ⍝ TODO: subsort by max
	g
}

Save←{
	⍝ ⍵: list of bitarrays to save
	⍝ ←: ⍬

	glyphs←⍵
	{
		n←⍕⍵
		g←⍵⊃glyphs
		⍝ ⎕←'writing' n
		name←n,'.gray'
		_←⎕SH 'touch ',name
		tie←name ⎕NTIE 0
		_←(-g) ⎕NREPLACE tie 0 83
		_←⎕NUNTIE tie
		_←⎕SH 'magick -depth 8 -size ',(' '⎕R'x'⍕⌽⍴g),' ',name,' ',n,'.png'
		_←⎕SH 'rm ',name
	}¨⍳≢glyphs
}

Loot←{
	⍝ ⍵: ⍬
	⍝ ←: list of curve sets each with shape 2 (x y) × 3 (start control end) × n (number of curves)

	⍝ ttx -s BQN386.ttf # for ttx files
	D2P←{
		d←⍵
		p←⍳≢d
		_←2{p[⍵]←⍺[⍺⍸⍵]}/⊂⍤⊢⌸d
		p
	}
	(d e x kv t)←↓⍉⎕XML⊃⎕NGET'font/BQN386._c_m_a_p.ttx'
	p←D2P d
	(u n)←↓⍉↑⊢/¨kv/⍨p=1⊃⍸p=1
	n←n[cs⍳⍨⎕UCS dec¨u]
	{
		(d e x kv t)←↓⍉⎕XML⊃⎕NGET'font/BQN386._g_l_y_f.',⍵,'.ttx'
		p←D2P d
		⊃,/{
			(x y on)←⍵
			m←2=/(⊢⍴⍨1+⍴)on
			i←⍸m
			x←(2÷⍨x[i]+x[(≢x)|i+1])@(1+(⊢+⍳⍤≢)i)⊢(m+1)/x
			y←(2÷⍨y[i]+y[(≢y)|i+1])@(1+(⊢+⍳⍤≢)i)⊢(m+1)/y
			on←(~on[i])@(1+(⊢+⍳⍤≢)i)⊢(m+1)/on ⍝ should be 1 0 1 0 ... 1 0
			x←⍉(⊢⍴⍨3,⍨3÷⍨⍴)1⌽(on+1)/x
			y←⍉(⊢⍴⍨3,⍨3÷⍨⍴)1⌽(on+1)/y
			↑x y
		}¨1↓kv{
			⊂↓⍉↑{(⊃1⊃⎕VFI)¨'-'⎕R'¯'¨⊢/⍵}¨⍵
		}⌸⍨p×p∊⍸'contour'∘≡¨e
	}¨'[A-Z]'⎕R'&_'¨n
}

Distribute←{
	⍝ ⍺: number of points to distribute
	⍝ ⍵: size of bins
	⍝ ←: ≢⍵ length vector, giving number of points in each bin
	l←⍵
	⊃{⍵@⍺⊢0⍴⍨≢l}/↓⍉{⍺,≢⍵}⌸1+(+\l)⍸⍺÷⍨(+/l)×⍳⍺
}

DistributeOverCurves←{
	⍝ ⍺: number of points to distribute
	⍝ ⍵: curve set with shape 2 (x y) × 3 (start control end) × n (curves)
	⍝ ←: (a b) where
	⍝    a: points matrix with shape 2 (x y) × ⍺
	⍝    b: angles vector with shape ⍺

	⍺←100
	⍝ https://raphlinus.github.io/curves/2018/12/28/bezier-arclength.html
	⍝ for now we do the very simple estimate
	a←  .5*⍨+⌿×⍨ -⌿⍤2⊢⍵[;2 0;] ⍝ length start → end
	b←+⌿.5*⍨+⌿×⍨2-⌿⍤2⊢⍵        ⍝ length start → control → end
	l←a+2×b                    ⍝ approx length
	n←⍺ Distribute l           ⍝ number of points on each curve roughly even, could be a way to do this that uses less space idk
	t←n+\⍤⍴¨÷n⌈n=0             ⍝ timesteps for each curve
	t←↑(1-t) t                 ⍝ and also complementary timesteps
	⍝             ┌─a─────────┐  ┌─b─────────┐ (new a and b)
	⍝ B(t) = (1-t)((1-t)P0+tP1)+t((1-t)P1+tP2)
	⍝         ┌────────│──┴│──────────│──┘│
	⍝         │     ┌──┴───┴──────────┴───┘
	⍝         ├──┐ ┌┴─┐
	a ←       +⌿⍤2⊢t×⍤2⊢⍵[;0 1;]
	b ←       +⌿⍤2⊢t×⍤2⊢⍵[;1 2;]
	xy←(⊃,/)⍤1+⌿⍤2⊢t×⍤2⊢1 0 2⍉↑a b
	⍝ B'(t) = 2(1-t)(P1-P0)+2t(P2-P1)
	m←Atan⌿+⌿⍤2(⊃,/)⍤1⊢t×⍤2⊢2×¯2-⌿⍤2⊢⍵
	xy m
}

EdgePoints←{
	⍝ ⍺: number of edgepoints to pick (default 100)
	⍝    if ⍺ > points in the image, there will be duplicate points
	⍝ ⍵: bit array to find edgepoints on
	⍝ ←: (a b) where
	⍝    a: points matrix with shape 2 (x y) × ⍺⋯)
	⍝    b: angles vector with shape ⍺

	⍺←100

	⍝ p←1 ¯1×⍤0 1⍉⌽↑⍸⍵∧⊃∨/⍵∘≠¨(1⊖⍵) (¯1⊖⍵) (1⌽⍵) (¯1⌽⍵)
	⍝ ⍺>1⊃⍴p: p
	⍝ p[;{⍵[⍋⍵]}⍺?1⊃⍴p]
	
	⍝ assumptions: we can always take a 3×3 window, and loops always close (no single-pixel wide chunks)
	b←⍵
	p←⍸⍵∧⊃∨/⍵∘≠¨(1⊖⍵) (¯1⊖⍵) (1⌽⍵) (¯1⌽⍵) ⍝ all edge points
	c←⍬                  ⍝ contours list
	ci←¯1+0×b            ⍝ contours list indices
	pi←2(⊢,¨⌽)1(⊢+⌽)3<⍳8 ⍝ indices of perimeter of the square
	_←{                  ⍝ for each point
		¯1≠⍵⌷ci: 1       ⍝ if this point has already been assigned a contour, continue
		⍺←{              ⍝ current contour (create if none)
			c,←⊂2 0⍴⍬    ⍝ add new contour
			¯1+≢c        ⍝ index in c of new contour
		}⍵
		(⍺⊃c),←⍪⍵        ⍝ add this point to the current contour
		(i j)←⍵
		ci[i;j]←⍺        ⍝ mark it on the matrix
		(i j)←⍵+¯1+pi⊃⍨8|1+⊃⍸2</9⍴b[¯1 0 1+i;¯1 0 1+j][pi] ⍝ indices of the next point
		ci[i;j]≠¯1: ⍺    ⍝ loop complete
		⍺∇i j            ⍝ continue on next point around the curve, tco babyy
	}¨p
	c←c{⍺[;⌊⍵÷⍨(1⊃⍴⍺)×⍳⍵]}¨⍺ Distribute (1⊃⍴)¨c ⍝ pick points from contour
	c←{0 (⌈/⍵[0;])+⍤0 1⊢1 ¯1×⍤0 1⊖⍵}¨c ⍝ i j → x y
	m←{
		n←1⊃⍴⍵
		n<3: n⍴0 ⍝ not enough points to get gradients
		Atan⌿-/⍵[;n|(⍳n)∘.+¯1 1]
	}¨c
	(⊃,/c) (⊃,/m)
}


ContextsPreprocess←{
	(rb tb)←⍺
	(xy a) ←⍵
	d←∘.-⍨⍤1⊢xy                        ⍝ x-x and y-y differences
	r←↓.5*⍨+⌿×⍨d                       ⍝ distances
	m←r>0                              ⍝ mask(s) of points that are different
	r←⍟m/¨r                            ⍝ filter r and log
	t←m/¨↓Atan2⌿d                      ⍝ angles (filtered)
	(max min)←(⌈/,⌊/)∊r
	(a r t m max min)                  ⍝─┐
}                                      ⍝ │
                                       ⍝ │
ContextsHistogram←{                    ⍝ │
	(rb tb)←⍺                          ⍝ │
	(a r t m max min)←⍵                ⍝←┘
	i   ←r⍸⍨¨⊂min+(max-min)×(⍳÷   ⊢)rb ⍝ bin distances
	i,¨¨←t⍸⍨¨⊂          ○¯1+(⍳÷.5×⊢)tb ⍝ bin angles
	F←{
		h←rb tb⍴⊂0 0                   ⍝ blank histogram
		h[⍺]+←↓⍵∘.(○⍨)1 2              ⍝ sum unit vectors in each bin
		(⊢÷.5*⍨+.×⍨)∊h                 ⍝ into vector and normalise
	}
	↑i F¨m/¨⊂a
}

Contexts←{
	⍝ ⍺[0]: number of distance bins
	⍝ ⍺[1]: number of angle bins
	⍝ ⍵: length m vector of pairs (a b) where
	⍝    a: points matrix with shape 2 (x y) × n
	⍝    b: angles vector with shape n
	⍝ ←: m×n×⍺[0]×⍺[1] array of shape contexts

	⍝ the contexts calculations are divided into two stages
	⍝ because we need to collect the min and max distances
	⍝ across all glyphs
	pp←⍺∘ContextsPreprocess¨⍵     ⍝ get the preprocessed values
	pp←(¯2↓¨pp),¨{(⌈/⍺),(⌊/⍵)}/↓⍉↑¯2↑¨pp ⍝ get max and min across all glyphs
	↑⍺∘ContextsHistogram¨pp       ⍝ continue
}

 
⍝ logging utils
time←¯1 12⎕DT⊂⎕TS
Log←{⎕←(30↑⍵) (1000÷⍨time-⍨¯1 12⎕DT⊂⎕TS)}

⍝ obtain shape contexts
np  ←100   ⍝ number of points to sample from the edges of a shape
bins←5 12  ⍝ number of radial and angle bins
input←⎕JSON⎕OPT'Dialect' 'JSON5'⊃⎕NGET'shape-contexts.json5' ⍝ input specification
cs←{
	⍵≡'lc':  'λ.()abcdefghijklmnopqrstuvwxyz'
	⍵≡'apl': '!#$%&''()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~˙∇⋄⍝⍺⍵¯×÷←↑→↓∆∊∘∧∨∩∪≠≡≢≤≥⊂⊃⊆⊖⊢⊣⊤⊥⌈⌊⌶⌷⎕⌸⌹⌺⌽⌿⍀⍉⍋⍎⍒⍕⍙⍟⍠⍣⍤⍥⍨⍪⍬⍱⍲⍳⍴⍷⍸○⍬⊇⍛⍢⍫√'
	⍝ (ascii - `"` + `˙` + apl/bqn specials)
}input.alphabet

⍝ (coords bearings) for each glyph
 fontPoints←np DistributeOverCurves¨ Loot ⍬
inputPoints←np EdgePoints¨           Split input.size Load input.path

VisualisePoints←{
	(xy m)←⍺⊃⍵
	py←'import matplotlib.pyplot as plt;'
	_←(⍉xy){py,←'plt.quiver(',(⍕0⊃⍺),',',(⍕1⊃⍺),',',(⍕2○⍵),',',(⍕1○⍵),');'}⍤1 0⊢m
	py,←'plt.show()'
	⎕SH 'python3 -c ''',('¯'⎕R'-'⊢py),''''
}

⍝ 0 VisualisePoints fontPoints
⍝ 0 VisualisePoints inputPoints

⍝ (nglyphs npoints bins[0] bins[1])
 font←bins Contexts  fontPoints
input←bins Contexts inputPoints

⍝ obtain shape contexts
nr←10               ⍝ number of representative points to select
ns←5                ⍝ number of font glyphs to shortlist
reps←input[;nr?np;] ⍝ filter to representative points

⍝                                               SHAPE                                    NOTE
c←↑reps∘.(↑∘.(.5*⍨+.×⍨⍤-)⍥(⊂⍤1))⍥(⊂⍤2)font ⍝ (ninputglyphs nfontglyphs nreps npoints) point-point matching costs
d  ←⌊/c                                    ⍝ (ninputglyphs nfontglyphs nreps)         smallest d_GSC
nu ←(≢font)÷⍨+⌿⍤2⊢d                        ⍝ (ninputglyphs nreps)                     normalisation factor N_u
d ÷←(≢font)⌿⍤2⊢nu⍴⍨(≢reps),1 nr            ⍝ (ninputglyphs nfontglyphs nreps)         normalised distances
k  ←⌊nr÷2                                  ⍝                                          number of RSCs to consider
d  ←k÷⍨+/{⍵[k↑⍋⍵]}⍤1⊢d                     ⍝ (ninputglyphs nfontglyphs)               representative distances between inputs and fonts
i  ←(⍋↑⍨ns⌊≢)⍤1⊢d                          ⍝ (ninputglyphs nshortlist)                shortlists for each input glyph

⍝ === DETAILED MATCHIING ON SHORTLIST ===
VisualiseMatching←{
	(j k)←⍺
	m←⍵[j;k;;]
	ip←⊃j⊃inputPoints
	ip÷←⌈/⌈/ip
	fp←⊃k⊃fontPoints[i[j;]]
	fp÷←⌈/⌈/fp
	py←'import matplotlib.pyplot as plt;'
	py,←∊{
		'plt.plot(',(∊ip[;⊃⍵]{'[',(⍕⍺),',',(⍕⍵),'],'}¨fp[;1⊃⍵]),'"g");'
	}¨⍸m
	py,←⊃{'plt.scatter(',⍺,',',⍵,', c = "red");'}⌿{'[',(∊',',¨⍨⍕¨⍵),']'}¨↓fp
	py,←⊃{'plt.scatter(',⍺,',',⍵,', c = "black");'}⌿{'[',(∊',',¨⍨⍕¨⍵),']'}¨↓ip
	py,←'plt.show()'
	⎕SH 'python3 -c ''',('¯'⎕R'-'⊢py),''''
}
⍝                                                                SHAPE                                     NOTE
c←{input[⍵;;](↑∘.(.5*⍨+.×⍨⍤-)⍥(⊂⍤1))⍤2⊢font[i[⍵;];;]}⍤0⍳≢input ⍝ (ninputglyphs nshortlist npoints npoints) point-point matching costs
m←assign⍤2⊢c                                                   ⍝ (ninputglyphs nshortlist npoints npoints) optimal assignments
⍝ {0 ⍵ VisualiseMatching m}¨⍳ns
c←+/+/c×m                                                      ⍝ (ninputglyphs nshortlist)                 glyph-glyph matching costs
⎕←⍉cs[i{⍺[⍋⍵]}⍤1⊢c]                                            ⍝ (nshortlist ninputglyphs)                 minimum cost assignments

⍝ TODO:
⍝ - [x] matching visualisations
⍝ - [x] consider difference in tangent angle between points
⍝ - [x] normalisation and weighting
⍝ - [x] fast pruning with representative contexts
⍝ - [x] round it out
⍝ - [x] nested matching cost calculations for less wsfulls
⍝ - [x] gsc: tangent in bins - doesn't work :wah:
⍝ - [x] lambda calc tests
⍝ - [x] get it installed in the handwriting server

⍝ FUTURE WORK:
⍝ - [ ] apl tests
⍝ - [ ] basic thin-plate splines
⍝ - [ ] regularised tps
⍝ - [ ] improve distribution of points over bezier curve
⍝ - [ ] more accurate angles from edgepoints (least squares?)
⍝ - [ ] draw from more fonts
⍝ - [ ] fast pruning with shapemes
⍝ - [ ] gaussian windows


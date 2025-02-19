⎕IO←0
⎕PW←12345

'dec' 'disp' 'displays'⎕CY'dfns'
'assign'⎕CY'dfns'
⍝ ascii - `"` + `˙` + `λ` + apl/bqn specials
⍝ cs←'!#$%&''()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~˙λ∇⋄⍝⍺⍵¯×÷←↑→↓∆∊∘∧∨∩∪≠≡≢≤≥⊂⊃⊆⊖⊢⊣⊤⊥⌈⌊⌶⌷⎕⌸⌹⌺⌽⌿⍀⍉⍋⍎⍒⍕⍙⍟⍠⍣⍤⍥⍨⍪⍬⍱⍲⍳⍴⍷⍸○⍬⊇⍛⍢⍫√'
cs←'λ.()abcdefghijklmnopqrstuvwxyz'
Atan2←12○⊣+0J1×⊢ ⍝ x Atan2 y | https://aplcart.info?q=atan2

input←⎕JSON⎕OPT'Dialect' 'JSON5'⊃⎕NGET'shape-contexts.json5'

Load←{
	⍝ ⍵: ⍬
	⍝ ←: bitarray
	tie←⍵ ⎕NTIE 0
	data←0≤(input.size⍴⎕NREAD tie 83 ¯1)
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
	⍝ ←: (a b c) where
	⍝    a: points matrix with shape 2 (x y) × ⍺
	⍝    b: angles vector with shape ⍺
	⍝    c: number of contours

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
	m←+∘(○0>⊢)⍨Atan2⌿+⌿⍤2(⊃,/)⍤1⊢t×⍤2⊢2×¯2-⌿⍤2⊢⍵ ⍝ normalised to >0
	xy m (2⊃⍴⍵)
}

Contexts←{
	⍝ ⍺[0]: number of distance bins
	⍝ ⍺[1]: number of angle bins
	⍝ ⍵: points matrix with shape 2 (x y) × n
	⍝ ←: n×⍺[0]×⍺[1] array of shape contexts

	⍺←5 12
	(rb tb)←⍺
	⍝ potential optimisation: only do triangle
	⍝ TODO: eliminate self
	d←-⍨⍤0 1⍨⍤1⊢⍵ ⍝ TODO: this can just be a ∘.-⍨⍤1 no?
	r←⍟(⊢∨0=⊢).5*⍨+⌿×⍨d
	t←Atan2⌿d
	i  ←1+r⍸⍨(⌈/∊r)×(⍳÷   ⊢)rb-1
	i,¨←  t⍸⍨○¯1+   (⍳÷.5×⊢)tb
	{
		h←rb tb⍴0
		h[⍵]+←1
		h
	}⍤1⊢i
}

EdgePoints←{
	⍝ ⍺: number of edgepoints to pick (default 100)
	⍝    if ⍺ > points in the image, there will be duplicate points
	⍝ ⍵: bit array to find edgepoints on
	⍝ ←: (a b c) where
	⍝    a: points matrix with shape 2 (x y) × ⍺⋯)
	⍝    b: angles vector with shape ⍺
	⍝    c: number of contours

	⍺←100

	⍝ ⍝ potential improvement: also check diagonal corners so we have more points
	⍝ p←1 ¯1×⍤0 1⍉⌽↑⍸⍵∧⊃∨/⍵∘≠¨(1⊖⍵) (¯1⊖⍵) (1⌽⍵) (¯1⌽⍵)
	⍝ ⍺>1⊃⍴p: p
	⍝ p[;{⍵[⍋⍵]}⍺?1⊃⍴p]
	
	⍝ assumptions: we can always take a 3×3 window, and loops always close (no single-pixel wide chunks)
	b←⍵
	p←⍸⍵∧⊃∨/⍵∘≠¨(1⊖⍵) (¯1⊖⍵) (1⌽⍵) (¯1⌽⍵) ⍝ all edge points
	c←⍬                  ⍝ contours list
	ci←¯1+0×b            ⍝ contours list indices
	pi←2(⊢,¨⌽)1(⊢+⌽)3<⍳8 ⍝ indices of perimeter of the square
	_←{
		⍺←{ ⍝ current contour (create if none)
			a←⍵⌷ci
			a≠¯1: a
			c,←⊂2 0⍴⍬
			¯1+≢c
		}⍵
		(⍺⊃c),←⍪⍵
		(i j)←⍵
		ci[i;j]←⍺
		(i j)←⍵+¯1+pi⊃⍨8|1+⊃⍸2</9⍴b[¯1 0 1+i;¯1 0 1+j][pi] ⍝ indices of the next point
		ci[i;j]≠¯1: ⍺ ⍝ loop complete
		⍺∇i j         ⍝ continue on next point around the curve, tco babyy
	}¨p
	c←c{⍺[;⌊⍵÷⍨(1⊃⍴⍺)×⍳⍵]}¨⍺ Distribute (1⊃⍴)¨c ⍝ pick points from contour
	c←{1 ¯1×⍤0 1⊖⍵}¨c ⍝ i j → x y
	m←{
		n←1⊃⍴⍵
		n<3: n⍴0 ⍝ not enough points to get gradients
		+∘(○0>⊢)⍨Atan2⌿-/⍵[;n|(⍳n)∘.+¯1 1]
	}¨c
	(⊃,/c) (⊃,/m) (≢c)
}

⍝ ⎕←'import matplotlib.pyplot as plt'
⍝ {
⍝ 	⎕←'plt.scatter('
⍝ 	⎕←'[0-9-,.]+'⎕R'[&],'⊢⍵⎕CSV⊂''
⍝ 	⎕←')'
⍝ 	⎕←'plt.show()'
⍝ }¨100 DistributeOverCurves¨ Loot ⍬

time←¯1 20⎕dt⊂⎕ts
Log←{⎕←(30↑⍵) (time-⍨¯1 20⎕DT⊂⎕TS)}

npoints←100
bins←5 12
sh←npoints,npoints,bins
font ←npoints DistributeOverCurves¨ Loot ⍬
input←npoints EdgePoints¨           Split Load input.path
⍝ Log 'done getting points'
Cost←{
	⍝ ⍞←'*'
	contexts←⍺{.5×+/+/(1 0 2 3⍉sh⍴⍺)(×⍨⍤-÷+)sh⍴⍵}⍥(bins∘Contexts⊃)⍵
	angles←|⍺∘.-⍥(1∘⊃)⍵
	c←0.9 0.1(⊃+.×)(⊢÷⌈/⍣2)¨contexts angles
	⍝ +/+/c×assign c ⍝ everybody say thank you John Scholes
	⍝ greedy matching (sacrifice accuracy for speed)
	+/{
		i←(⊢⍳⌊/),c
		j←⌊i÷1⊃⍴c
		k←i|⍨1⊃⍴c
		w←c[j;k]
		c⌿⍨←~(0⊃⍴c)↑⍸⍣¯1,j
		c/⍨←~(1⊃⍴c)↑⍸⍣¯1,k
		w
	}¨⍳≢c
}
⍝ cost←.1 .9(⊃+.×)(⊢÷⌈/⍣2)¨(|input∘.-⍥(2∘⊃¨)font)(input ∘.Cost font)
cost←input ∘.Cost font
⍝ Log 'done matching'

⍝ (⍪'{(+⌿⍵)÷≢⍵}'),' ',
⎕←⍉cs[(10↑⍋)⍤1⊢cost]

⍝ TODO:
⍝ - [ ] matching visualisations
⍝ - [x] consider difference in tangent angle between points
⍝ - [x] normalisation and weighting
⍝ - [ ] improve distribution of points over bezier curve
⍝ - [ ] look at thin plate spline shit
⍝ - [ ] lambda calc tests
⍝ - [ ] more accurate angles from edgepoints (least squares)


from fpdf import FPDF
import random
import itertools

# Set seed for reproducibility (optional)
random.seed(42)

def generate_linear():
    eqs = set()
    while len(eqs) < 200:
        a = random.choice([1,2,3,4,5,6,7,8,9,10])
        b = random.randint(-15, 15)
        sol = random.randint(-10, 10)
        c = a * sol + b
        eq = f"{a}x + {b} = {c}"
        eqs.add(eq)
    return list(eqs)

def generate_rational():
    eqs = set()
    while len(eqs) < 200:
        a = random.randint(1, 5)
        b = random.randint(-10, 10)
        c = random.randint(1, 5)
        d = random.randint(-8, 8)
        if c == 0:
            continue
        k_options = [0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, -0.5, -1, -1.5, -2]
        k = random.choice(k_options)
        
        denominator = a - (k * c)
        if denominator == 0:
            continue
        x = (k * d - b) / denominator
        
        if abs(x) < 20 and x != -d/c and x != 0:
            if c == 1:
                eq = f"({a}x + {b})/(x + {d}) = {k}"
            else:
                eq = f"({a}x + {b})/({c}x + {d}) = {k}"
            eqs.add(eq)
    return list(eqs)

def generate_quadratic():
    eqs = set()
    while len(eqs) < 200:
        r1 = random.randint(-12, 12)
        r2 = random.randint(-12, 12)
        if r1 == r2:
            continue
        
        a = 1
        b = -(r1 + r2)
        c = r1 * r2
        
        if random.choice([True, False]) and abs(b) < 20:
            scale = random.choice([2, 3, -1])
            a, b, c = a*scale, b*scale, c*scale
        
        if b >= 0 and c >= 0:
            eq = f"{a}x^2 + {b}x + {c} = 0"
        elif b >= 0 and c < 0:
            eq = f"{a}x^2 + {b}x - {abs(c)} = 0"
        elif b < 0 and c >= 0:
            eq = f"{a}x^2 - {abs(b)}x + {c} = 0"
        else:
            eq = f"{a}x^2 - {abs(b)}x - {abs(c)} = 0"
        
        eqs.add(eq)
    return list(eqs)

def generate_inequalities():
    ineqs = set()
    symbols = [">", "<", ">=", "<="]
    while len(ineqs) < 200:
        a = random.randint(1, 8)
        b = random.randint(-15, 15)
        sym = random.choice(symbols)
        sol = random.randint(-12, 12)
        c = a * sol + b
        
        if random.choice([True, False]):
            eq = f"{a}x + {b} {sym} {c}"
        else:
            if random.choice([True, False]) and sym in [">", "<"]:
                shift = random.randint(1, 5)
                left = c - shift
                right = c + shift
                eq = f"{left} {sym} {a}x + {b} {sym} {right}"
            else:
                eq = f"{a}x + {b} {sym} {c}"
        
        ineqs.add(eq)
    return list(ineqs)

print("Generating 200 random linear equations...")
linear_eqs = generate_linear()

print("Generating 200 random rational equations...")
rational_eqs = generate_rational()

print("Generating 200 random quadratic equations...")
quadratic_eqs = generate_quadratic()

print("Generating 200 random inequalities...")
inequality_eqs = generate_inequalities()

# Create PDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=8)

categories = {
    "Linear Equations": linear_eqs,
    "Rational Equations": rational_eqs,
    "Quadratic Equations": quadratic_eqs,
    "Inequalities (Linear)": inequality_eqs
}

for title, eq_list in categories.items():
    pdf.set_font("Arial", 'B', 11)
    pdf.cell(0, 8, title, ln=True)
    pdf.set_font("Arial", size=7)
    
    # 2-column layout
    col_width = 95
    x_start = pdf.get_x()
    
    for i, eq in enumerate(eq_list, 1):
        if i % 2 == 1:
            pdf.set_x(x_start)
        else:
            pdf.set_x(x_start + col_width)
        
        pdf.cell(col_width, 5, f"{i:3d}. {eq}", ln=False)
        
        if i % 2 == 0:
            pdf.ln(5)
        
        if i % 100 == 0:
            pdf.add_page()
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 8, title + f" (continued)", ln=True)
            pdf.set_font("Arial", size=7)
            x_start = pdf.get_x()
    
    pdf.add_page()

# Save locally (same folder as script)
pdf.output("math_equations_randomized.pdf")
print("\n✅ PDF generated: math_equations_randomized.pdf")
print(f"📊 Total equations: {len(linear_eqs) + len(rational_eqs) + len(quadratic_eqs) + len(inequality_eqs)}")
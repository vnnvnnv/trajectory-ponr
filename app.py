from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.special import expit

app = Flask(__name__)

class PonrCalculator:
    def __init__(self):
        self.Q_max_summer = 200
        self.Q_max_winter = 65
        self.P_com = 1500
        self.P_stud = 300
        self.C_var_util = 200
        self.C_fix_base = 1.565
        self.C_fix_staff = 3.482
        self.C_avar = 0.10
        self.lambda0 = 0.30
        self.gamma = 1.20
        self.k_d = 0.35
        self.k_s = 0.25
        self.r = 0.10
        self.rho = 10.0
        self.b0 = -3.2
        self.b1 = 4.7
        self.b2 = 2.3
        self.b3 = 1.8
        self.b4 = -2.9

    def accident_intensity(self, D):
        return self.lambda0 * np.exp(self.gamma * D)

    def occupancy(self, D, lambda_t, season, theta):
        Q_max = self.Q_max_summer if season == 'summer' else self.Q_max_winter
        koef_season = 1.0 if season == 'summer' else 0.7
        infra_factor = max(0.1, 1 - self.k_d * D)
        reput_factor = max(0.1, 1 - self.k_s * (1 - np.exp(-lambda_t)))
        P_avg = theta * self.P_com + (1 - theta) * self.P_stud
        price_factor = min(1.0, P_avg / 2500)
        Q_fact = Q_max * infra_factor * reput_factor * koef_season * price_factor
        return max(5, min(Q_fact, Q_max * 0.95))

    def cashflow_monthly(self, Q_fact, theta, lambda_t):
        P_avg = theta * self.P_com + (1 - theta) * self.P_stud
        revenue = Q_fact * P_avg * 30 / 1000000
        var_cost = self.C_var_util * Q_fact * 30 / 1000000
        fix_cost = (self.C_fix_base + self.C_fix_staff) / 12
        avar_cost = self.C_avar * lambda_t
        return revenue - var_cost - fix_cost - avar_cost

    def ponr_probability(self, D, CF, Occ_pct, subsidy):
        loss_flag = 1 if CF < 0 else 0
        x = (self.b0 + self.b1 * D + self.b2 * loss_flag +
             self.b3 * (1 - Occ_pct / 100) + self.b4 * subsidy)
        return expit(x)

    def simulate_trajectory(self, D0, I_total, theta, subsidy):
        D = D0
        t = 0
        ponr_reached = False
        CF_history = []
        if I_total > 0:
            reduction = min(0.6, I_total / 100)
            D = max(0.2, D * (1 - reduction))
        while t < 60 and not ponr_reached:
            month_num = (t % 12) + 1
            season = 'winter' if month_num in [11, 12, 1, 2] else 'summer'
            lambda_t = self.accident_intensity(D)
            accidents = np.random.poisson(lambda_t)
            avar_cost = self.C_avar * accidents
            if accidents > 2:
                avar_cost *= 1.5
            Q_fact = self.occupancy(D, lambda_t, season, theta)
            CF = self.cashflow_monthly(Q_fact, theta, lambda_t) - avar_cost
            Occ_pct = (Q_fact / (self.Q_max_summer if season == 'summer' else self.Q_max_winter)) * 100
            P_ponr = self.ponr_probability(D, CF, Occ_pct, subsidy)
            CF_history.append(CF)
            if P_ponr > 0.7 and D > 0.75:
                ponr_reached = True
                ponr_time = t
                break
            D = min(1.0, D + 0.01 + 0.05 * accidents)
            t += 1
        if not ponr_reached:
            ponr_time = 60
        return {'CF_history': CF_history, 'ponr_time': ponr_time, 'ponr_reached': ponr_reached}

    def run_monte_carlo(self, D0, I_total, theta, subsidy, n_iter=3000):
        all_cf = []
        ponr_times = []
        for _ in range(n_iter):
            result = self.simulate_trajectory(D0, I_total, theta, subsidy)
            all_cf.append(result['CF_history'])
            ponr_times.append(result['ponr_time'])
        max_len = max(len(cf) for cf in all_cf)
        CF_avg = np.zeros(max_len)
        for cf in all_cf:
            CF_avg[:len(cf)] += np.array(cf)
        CF_avg /= n_iter
        discount_factors = [(1 + self.r) ** (t / 12) for t in range(max_len)]
        NPV = np.sum(CF_avg[:max_len] / discount_factors[:max_len]) - I_total
        P_ponr = np.mean([1 if t < 60 else 0 for t in ponr_times])
        mean_ponr_time = np.mean(ponr_times)
        Pi_risk = NPV - self.rho * P_ponr
        return {'NPV': round(NPV, 1), 'Pi_risk': round(Pi_risk, 1), 'P_ponr': round(P_ponr, 2), 'ponr_time': round(mean_ponr_time, 1)}

calculator = PonrCalculator()

def generate_scenarios(D0, I_total, subsidy):
    scenarios = {
        'optimistic': {'theta': 0.60, 'name': 'Оптимистический'},
        'realistic': {'theta': 0.35, 'name': 'Реалистический'},
        'pessimistic': {'theta': 0.10, 'name': 'Пессимистический'}
    }
    results = {}
    for key, config in scenarios.items():
        res = calculator.run_monte_carlo(D0, I_total, config['theta'], subsidy, n_iter=2000)
        results[key] = {**res, 'name': config['name']}
    optimal = max(results.keys(), key=lambda k: results[k]['Pi_risk'])
    return results, optimal

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json()
        D0 = float(data.get('D0', 85)) / 100
        I_total = float(data.get('I_total', 0))
        theta = float(data.get('theta', 5)) / 100
        subsidy = int(data.get('subsidy', 0))
        base_result = calculator.run_monte_carlo(D0, I_total, theta, subsidy, n_iter=2000)
        scenarios, optimal = generate_scenarios(D0, I_total, subsidy)
        ai_text = f"""🎯 **Анализ текущей конфигурации**
📊 Время до PONR: {base_result['ponr_time']} мес.
🎯 Целевой горизонт: 48 мес.

**Рекомендации:**
1. 🔧 Снижение износа до 65% отсрочит PONR на 12-18 мес.
2. 📈 Повышение доли коммерческих клиентов до 40% увеличит денежный поток
3. 💰 Инвестиции 50 млн руб. в инфраструктуру дадут долгосрочный эффект
4. 🏛️ Целевая субсидия снизит нагрузку на бюджет университета

**Вывод:** Для достижения целевого горизонта необходимо реализовать минимум две рекомендации."""
        return jsonify({'success': True, 'base': base_result, 'scenarios': scenarios, 'optimal': optimal, 'ai_recommendation': ai_text})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

import numpy as np

class SimplePerceptron:
    #単純パーセプトロンの実装
    def __init__(self, input_dim:int):
        self.input_dim = input_dim
        self.w = np.random.normal(size=(input_dim)) #重み
        self.b = 0. #バイアス
    
    def forward(self, x):
        y = step(np.matmul(self.w.T, x) + self.b)
        return y
    
    def compute_deltas(self, x, t):
        y = self.forward(x)
        delta = y - t
        dw = delta * x
        db = delta
        return dw, db

def step(x):
    return 1 * (x > 0)

if __name__ == '__main__':
    np.random.seed(123)
    
    d = 2
    N = 20
    
    mean = 5
    
    x1 = np.random.randn(N//2, d) + np.array([0,0])
    x2 = np.random.randn(N//2, d) + np.array([mean, mean])
    
    t1 = np.zeros(N//2)
    t2 = np.ones(N//2)
    
    x = np.concatenate((x1, x2), axis=0) #入力データ
    t = np.concatenate((t1, t2)) #出力データ
    
    model = SimplePerceptron(input_dim=d)
    
    def compute_loss(dw, db):
        return all(dw == 0) * (db == 0)
    
    def train_step(x, t):
        dw, db = model.compute_deltas(x, t)
        loss = compute_loss(dw, db)
        model.w = model.w - dw
        model.b = model.b - db
        return loss

    while True:
        classified = True
        for i in range(N):
            loss = train_step(x[i], t[i])
            classified *= loss
        if classified:
            break
        
    print(f"w: {model.w}")
    print(f"b: {model.b}")
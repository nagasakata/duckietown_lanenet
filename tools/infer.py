class Function():

    def __init__(self, coefficient_list):
        self.a, self.b, self.c = coefficient_list[0], coefficient_list[1], coefficient_list[2]

        self.vertex_x = - self.b / (2 * self.a)
        self.vertex_y = - ((self.b ** 2) / (4 * self.a)) + self.c

        self.diff_a = 2 * self.a
        self.diff_b = self.b
        

    def get_vertex(self):

        return self.vertex_x, self.vertex_y

    def differential(self):

        return self.diff_a, self.diff_b

    def get_inclination_by_x(self, x):

        return self.diff_a * x + self.diff_b
    
    def get_inclination_by_y(self, y):

        if self.a < 0 and y > self.vertex_y:
            print("the vertex is under the y you set")
            diff_1, diff_2 = None, None
        elif self.a > 0 and y < self.vertex_y:
            print("the vertex is upper the y you set")
            diff_1, diff_2 = None, None
        else:
            x1 = (-self.b + ((self.b ** 2 - 4 * self.a * (self.c - y) ) ** 0.5)) / (2 * self.a)
            x2 = (-self.b - ((self.b ** 2 - 4 * self.a * (self.c - y) ) ** 0.5)) / (2 * self.a)
            diff_1 = self.get_inclination_by_x(x1)
            diff_2 = self.get_inclination_by_x(x2)

        return x1, x2, diff_1, diff_2
    
    def solve


now_func = Function([1.0, 2.0, 3.0])

print(now_func.get_vertex())
print(now_func.get_inclination_by_x(1))
print(now_func.get_inclination_by_y(6))
from pulp import *
from util import time, Capturing
from termcolor import colored

"""
## Zone Problem ##

### General Characteristics ###

- Motorway only at straights
- 30 Zones and No Passing can overlap
- Motorway can only be in urban
- No Passing can be in both. Can there be no passing in transition between urban and rural?

### Input ###

 - IP config
 - Zone config
 
### Procedure ###

1. Before Solving IP

- Use Zone config to extend/alter IP config
    - Parking => At least one Straight of length 20m or more
    - Motorway => At least one Straight of length 10m or more
- Use new combined Straights constraint for this
- e.g. Minimum 2 straights of length 2 or more AND Minimum one straight of length 4 or more
- Possible Solution: One Straight of length 2, One Straight of length 4

2. Solve IP
- Get Solution and dict
- Convert Solution to graph and get graph tour
- Identify Straights from step 1 and assign them their correct zones (Parking and Motorway)
- Save lengths of all remaining gaps as list

3. Solve 30 Zone Problem
- For each gap:
-> Make new vars: gap_1, zone_1, ..., gap_n, zone_n
-> n depends on the sum of the minimum lengths of the requested 30 Zones, sorted in ascending order
   If the 3 smallest zones can fit, we need at least 3 zone vars
-> Solve this amazing problem
 
4. Solve No Passing Zone Problem
- For each gap:
-> See if there is space for no passing. HOWEVER: No passing may not start or end, where 30 Zones start/end


### Problems ###

- What if requested 30 and no passing zones can no longer be placed?
- Is real integration into ip neccessary?
- How could it be done?
- What about iteration? Prohibtion no longer functional? Unless zones are optional constraints
- 
"""


class ZoneProblem:
    def __init__(self, zone_descriptions, blocked_zones, n, min_gap=2):
        # print("ZONE IP PROBLEM:\ndesc: {}\nblocked: {}\nn={}\ngap={}".format(zone_descriptions, blocked_zones, n, min_gap))
        self.zone_descriptions = zone_descriptions
        self.blocked_zones = blocked_zones
        self.n = n
        self.min_gap = min_gap
        self.zone_pointers, self.zone_requirements = self.init_variables()
        self.problem = LpProblem("ZoneProblem", LpMinimize)
        self.add_base_constraints()
        self.add_requirements_constraints()
        self.add_blocked_constraints()

    def init_variables(self):
        zone_pointers = []
        zone_requirements = []

        for i, _ in enumerate(self.zone_descriptions):
            start = LpVariable("{}_{}".format("start", i), lowBound=0, upBound=self.n, cat=const.LpInteger)
            end = LpVariable("{}_{}".format("end", i), lowBound=0, upBound=self.n, cat=const.LpInteger)
            zone_pointers.append((start, end))
            requirements = []
            for j, _ in enumerate(self.zone_descriptions):
                mic = LpVariable("{}_{}_{}".format("mic", i, j), cat=const.LpBinary)
                mac = LpVariable("{}_{}_{}".format("mac", i, j), cat=const.LpBinary)
                r = LpVariable("{}_{}_{}".format("r", i, j), cat=const.LpBinary)
                requirements.append((mic, mac, r))
            zone_requirements.append(requirements)

        return zone_pointers, zone_requirements

    def add_base_constraints(self):
        # end must follow start
        for (start, end) in self.zone_pointers:
            self.problem += start + 1 <= end

        # # zones must follow each other
        for i in range(len(self.zone_pointers)-1):
            start1, end1 = self.zone_pointers[i]
            start2, end2 = self.zone_pointers[i+1]
            self.problem += end1 + 1 + self.min_gap <= start2

    def add_requirements_constraints(self):
        n = self.n
        for i, (start, end) in enumerate(self.zone_pointers):
            requirements_i = []
            for j, (mic, mac, r) in enumerate(self.zone_requirements[i]):
                min_j, max_j = self.zone_descriptions[j]
                # Zone i fullfils minimum length from requirement j
                self.problem += mic <= 1 + ((end - start) - min_j) / n
                self.problem += mic >= ((end - start) - min_j) / n
                # Zone i fullfils maximum length from requirement j
                self.problem += mac <= 1 + (max_j - (end - start)) / n
                self.problem += mac >= (max_j - (end - start)) / n
                # Zone i fullfils requirement j (both min and max)
                self.problem += r <= (mic + mac) / 2
                # IMPORTANT: the below backwards direction must not be true! This way the single requirement per zone constraint can work
                # self.problem += r >= mic + mac - 1
                requirements_i.append(r)

            # Each Zone must fullfil a single requirement
            self.problem += sum(requirements_i) == 1

        for j in range(len(self.zone_descriptions)):
            all_requirements_for_description_j = []
            for i in range(len(self.zone_requirements)):
                mic, mac, r = self.zone_requirements[i][j]
                all_requirements_for_description_j.append(r)

            # Each zone description must be satisfied exactly once
            self.problem += sum(all_requirements_for_description_j) == 1

    def add_blocked_zone_constraints(self):
        n = self.n
        for i, (start, end) in enumerate(self.zone_pointers):
            requirements_i = []
            for j, (mic, mac, r) in enumerate(self.zone_requirements[i]):
                min_j, max_j = self.zone_descriptions[j]
                # Zone i fullfils minimum length from requirement j
                self.problem += mic <= 1 + ((end - start) - min_j) / n
                self.problem += mic >= ((end - start) - min_j) / n
                # Zone i fullfils maximum length from requirement j
                self.problem += mac <= 1 + (max_j - (end - start)) / n
                self.problem += mac >= (max_j - (end - start)) / n
                # Zone i fullfils requirement j (both min and max)
                self.problem += r <= (mic + mac) / 2
                # IMPORTANT: the below backwards direction must not be true! This way the single requirement per zone constraint can work
                # self.problem += r >= mic + mac - 1
                requirements_i.append(r)

            # Each Zone must fullfil a single requirement
            self.problem += sum(requirements_i) == 1

        for j in range(len(self.zone_descriptions)):
            all_requirements_for_description_j = []
            for i in range(len(self.zone_requirements)):
                mic, mac, r = self.zone_requirements[i][j]
                all_requirements_for_description_j.append(r)

            # Each zone description must be satisfied exactly once
            self.problem += sum(all_requirements_for_description_j) == 1

    def add_blocked_constraints(self):
        n = self.n
        for i, (start, end) in enumerate(self.zone_pointers):
            for b, (start_b, end_b) in enumerate(self.blocked_zones):
                # Substract/Add gap + 1 to before and after to prevent directly adjacent zones
                start_b -= (self.min_gap + 1)
                end_b += (self.min_gap + 1)

                # New var 'before' that is 1, if zone i is before blocked area b, 0 else
                before = LpVariable("{}_{}_{}".format("before", i, b), cat=const.LpBinary)
                self.problem += before <= 1 + (start_b - end) / n
                self.problem += before >= (start_b - end) / n

                # New var 'after' that is 1, if zone i is after blocked area b, 0 else
                after = LpVariable("{}_{}_{}".format("after", i, b), cat=const.LpBinary)
                self.problem += after <= 1 + (start - end_b) / n
                self.problem += after >= (start - end_b) / n

                # New var 'no_overlap' that is 1, if zone i does not overlap with blocked area b, 0 else
                no_overlap = LpVariable("{}_{}_{}".format("no_overlap", i, b), cat=const.LpBinary)
                self.problem += no_overlap <= before + after
                self.problem += no_overlap >= before + after

                # force no overlap
                self.problem += no_overlap == 1

    def show_solution(self):
        for i, (start, end) in enumerate(self.zone_pointers):
            requirements_i = []
            for j, (mic, mac, r) in enumerate(self.zone_requirements[i]):
                if value(r) > 0:
                    print("Zone {} with pointers ({}, {}) satisifies requirement {} = {}".format(i, colored(int(value(start)), 'yellow'), colored(int(value(end)), 'yellow'), j, self.zone_descriptions[j]))
                requirements_i.append(r)

            # Each Zone must fullfil a single requirement
            self.problem += sum(requirements_i) == 1

    def solve(self, _print=False):
        try:
            with Capturing() as output:
                status = self.problem.solve(GUROBI(msg=0))
        except:
            print(colored('GUROBI IS NOT AVAILABLE. DEFAULTING TO CBM!', 'red'))
            status = self.problem.solve(PULP_CBC_CMD(msg=0))

        solution = [(value(start), value(end)) for (start, end) in self.zone_pointers]

        if _print:
            print("{} Solution: {}".format(LpStatus[status], solution))
        return solution, status + 1


if __name__ == '__main__':
    p = ZoneProblem(zone_descriptions=[(2, 4), (5, 5), (4, 10)], blocked_zones=[(0, 4), (10, 13)], n=30)
    _start = time.time()
    solution, status = p.solve()
    _end = time.time()
    print(colored("Solution {}, Time elapsed: {:.2f}s".format(LpStatus[status - 1], _end - _start), "green" if status > 1 else "red"))
    if status > 0:
        p.show_solution()

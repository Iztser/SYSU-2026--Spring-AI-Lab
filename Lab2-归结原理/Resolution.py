import time

variables = {'x','y','z','u','v','w','xx','yy','zz','uu','vv','ww'}


#0:变量 1:常量 2:函数/谓词
def classify_item(s:str):
    if s in variables:
        return 0
    if '(' not in s:
        return 1
    return 2

def unify(term1, term2, subst = None):
    if subst is None:
        subst = {}

    if term1 in subst:
        return unify(subst[term1], term2, subst)
    if term2 in subst:
        return unify(term1, subst[term2],subst)

    if classify_item(term1) == 0 and classify_item(term2) == 0:
        subst[term1] = term2
        return True,subst
        
    if classify_item(term1) == 0:
        if term1 in term2:
            return False, subst
        subst[term1] = term2
        return True,subst
        
    if classify_item(term2) == 0:
        if term2 in term1:
            return False, subst
        subst[term2] = term1
        return True, subst
        
    return term1 == term2, subst
    
def unify_literals(lit1,lit2):
    def parse(lit):
        isneg = lit.startswith('~')
        body = lit[1:] if isneg else lit
        pred = body[:body.index('(')]
        args = body[body.index('(') + 1:-1].split(',')
        return isneg, pred, args
    
    isneg1, pred1, args1 = parse(lit1)
    isneg2, pred2, args2 = parse(lit2)

    if pred1 != pred2 or isneg1 == isneg2:
        return False,{}
    
    if len(args1) != len(args2):
        return False, {}
    
    subst = {}
    for a1, a2 in zip(args1, args2):
        ok, subst = unify(a1, a2, subst)
        if not ok:
            return False, {}
    return True, subst
    
def apply_subst(clause, subst):
    res = []
    for lit in clause:
        isneg = lit.startswith('~')
        body = lit[1:] if isneg else lit
        pred = body[:body.index('(')]
        args = body[body.index('(') + 1:-1].split(',')
        new_args = [subst.get(a, a) for a in args]
        new_lit = f"{pred}({','.join(new_args)})"
        if isneg:
            new_lit = '~' + new_lit
        res.append(new_lit)
    return tuple(res)

def resolve(clause1, clause2):
    clause1 = list(clause1)
    clause2 = list(clause2)

    for i, lit1 in enumerate(clause1):
        for j, lit2 in enumerate(clause2):
            ok, subst = unify_literals(lit1, lit2)
            if ok:
                new1 = [l for k, l in enumerate(clause1) if k != i]
                new2 = [l for k, l in enumerate(clause2) if k != j]
                new_clause = apply_subst(new1 + new2, subst)
                return True, new_clause, subst, (i,j)
    return False, (), {}, (-1, -1)

def backsearch(resolution_step, N):
    visited = set()
    st = []

    def trace(clause_id):
        if clause_id < N or clause_id in visited:
            return
        visited.add(clause_id)
        step = resolution_step[clause_id]
        st.append(step)
        for p in step["parents"]:
            trace(p)

    trace(len(resolution_step) - 1)
    return resolution_step[:N] + st[::-1]
    
def renumber_ids(result, N):
    all_ids = {it["id"] for it in result[N:]}
    id_map = {old: N + i for i, old in enumerate(sorted(all_ids))}
    id_map.update({i:i for i in range(N)})

    renumbered = []
    for item in result[N:]:
        new_item = {
            "id":id_map[item["id"]],
            "clause":item["clause"],
            "parents":[id_map[p] for p in item["parents"]],
            "letters":item["letters"],
            "replacement":item["replacement"]
        }
        renumbered.append(new_item)
    return result[:N] + sorted(renumbered, key=lambda x: x["id"])


def Resolution(KB):
    KB = list(KB)
    N = len(KB)
    resolution_step = []
    clause_set = set()
    max_len = max(len(c) for c in KB)

    for idx, clause in enumerate(KB):
        resolution_step.append({
            "id":idx,
            "clause":clause,
            "parents":[],
            "letters":[],
            "replacement":{}
        })
        clause_set.add(clause)

    current_id = N
    proved = False
    
    while True:
        new_clauses = []
        for i in range(len(resolution_step)):
            for j in range(i + 1,len(resolution_step)):
                clause1 = resolution_step[i]["clause"]
                clause2 = resolution_step[j]["clause"]

                ok, res_clause, subst, (liti,litj) = resolve(clause1,clause2)
                if not ok:
                    continue

                #得到空语句，证明成功
                if len(res_clause) == 0:
                    resolution_step.append({
                        "id":current_id,
                        "clause":res_clause,
                        "parents":[i,j],
                        "letters":([liti], [litj]),
                        "replacement":subst
                    })
                    proved = True
                    break

                if res_clause not in clause_set and len(res_clause) <= max_len:
                    clause_set.add(res_clause)
                    new_clauses.append((current_id,res_clause,[i,j],([liti], [litj]),subst))
                    current_id += 1
            if proved:
                break
        if proved:
            break
        if not new_clauses:
            break
        for item in new_clauses:
            resolution_step.append({
                "id":item[0],
                "clause":item[1],
                "parents":item[2],
                "letters":item[3],
                "replacement":item[4]
            })

    result = backsearch(resolution_step,N)
    result = renumber_ids(result,N)

    output = []
    for idx,item in enumerate(result):
        clause_id = idx + 1
        clause = item["clause"]
        if not item["parents"]:
            line = f"{clause_id}({','.join(clause)})"
        else:
            p1, p2 = item["parents"]
            li, lj = item["letters"]
            subst = item["replacement"]
            rule = f"R[{p1 + 1}{chr(ord('a') + li[0])},{p2 + 1}{chr(ord('a') + lj[0])}]"
            if subst:
                s = ",".join([f"{k}={v}" for k, v in subst.items()])
                rule += f"{{{s}}}"
            line = f"{clause_id}{rule} = ({','.join(clause)})"
        output.append(line)
    return output, proved


    
if __name__ == "__main__":
    KB1 = { ("A(mike)",), ("A(tony)",), ("A(john)",), ("L(tony,rain)",), ("L(tony,snow)",),
       ("~A(x)", "S(x)", "C(x)"), ("~C(y)", "~L(y,rain)"), ("L(z,snow)", "~S(z)"),
       ("~L(tony,u)", "~L(mike,u)"), ("L(tony,v)", "L(mike,v)"), ("~A(w)", "~C(w)", "S(w)")}

    KB2 = {
    ("On(tony,mike)",),
    ("On(mike,john)",),
    ("Green(tony)",),
    ("~Green(john)",),
    ("~On(xx,yy)", "~Green(xx)", "Green(yy)")}

    
    start1_time = time.perf_counter()
    res1, ok1 =Resolution(KB1)
    for line in res1:
        print(line)
    sum(range(1000000))

    end1_time = time.perf_counter()

    print(f"运行时间：{end1_time - start1_time:.6f} 秒")

    
    start2_time = time.perf_counter()
    res2, ok2 =Resolution(KB2)
    for line in res2:
        print(line)
    sum(range(1000000))

    end2_time = time.perf_counter()

    print(f"运行时间：{end2_time - start2_time:.6f} 秒")

import json, csv, random, collections, sys
csv.field_size_limit(10**8)
K=['acl','mcl','mm','lm','moa','loa','pfoa','eff','syn','bak','con','frx']
NAME=dict(zip(K,["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]))
lab={}
for l in open(sys.argv[1],encoding='utf-8'):
    d=json.loads(l); lab[d['StudyInstanceUID']]=d
rows=list(csv.DictReader(open(sys.argv[2],encoding='utf-8')))
rep={r['StudyInstanceUID']:r['Report'] for r in rows}
gold={r['StudyInstanceUID']:r for r in rows if r['ACL'].strip()!=''}
# gold cross-check: na cells on gold studies
g=collections.Counter()
for s,r in gold.items():
    for k in K:
        if lab[s][k+'_v']=='na': g[(k,int(r[NAME[k]].strip()=='1'))]+=1
print('GOLD na cells (finding: gold_neg, gold_pos):', {k:(g[(k,0)],g[(k,1)]) for k in K})
rng=random.Random(7); used=set(); out=[]
for k in K:
    cand=[s for s in lab if lab[s][k+'_v']=='na' and s not in used and s not in gold]
    rng.shuffle(cand)
    for s in cand[:7]:
        used.add(s); out.append((k,s))
with open(sys.argv[3],'w',encoding='utf-8') as f:
    for i,(k,s) in enumerate(out,1):
        d=lab[s]; others=' '.join(f"{x}={d[x+'_v']}" for x in K)
        f.write(f"### {i:02d} | FINDING={NAME[k]} | p={d[k+'_p']} | study=...{s[-10:]}\nother stances: {others}\n{rep[s].strip()}\n\n")
print('wrote',len(out),'cells')

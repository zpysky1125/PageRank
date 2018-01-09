import com.google.common.collect.Lists;
import org.apache.commons.lang.StringUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;


public class sparkPageRankAverageV2 {
    public static void main(String[] args) throws Exception{
        if(args.length < 3) {
            System.out.println("Error Input");
            System.exit(1);
        }
        long start = System.currentTimeMillis();
        String input = args[0], output = args[1];
        Double epison = Double.parseDouble(args[2]), rankDif;
        SparkConf conf = new SparkConf().setAppName("sparkPageRank");
        JavaSparkContext jsc = new JavaSparkContext(conf);

        JavaRDD<String> lines = jsc.textFile(input);
        JavaPairRDD<String, String> pairs = lines
                .mapToPair(line -> {
                    String[] lineSplit = line.split("\t");
                    return new Tuple2<>(StringUtils.strip(lineSplit[0], "\""), StringUtils.strip(lineSplit[1], "\""));
                })
                .filter(p->(!p._1.equals("0") && !p._2.equals("0"))).distinct();


        JavaRDD<String> fromSet = pairs.keys().distinct();
        JavaRDD<String> toSet = pairs.values().distinct();
        JavaRDD<String> vertex = fromSet.union(toSet).distinct();

        // 如果一个点没有出边，增加一个到自己的出边，以此保证在矩阵乘法过程中，不会出现 rank 的弥散
        JavaPairRDD<String, Iterable<String>> graphTmp = vertex.subtract(fromSet).mapToPair(v->new Tuple2<>(v,"none")).union(pairs).groupByKey().cache();

        JavaPairRDD<String, Iterable<String>> graph = toSet.mapToPair(v->new Tuple2<>(v, "")).join(graphTmp).mapToPair(v->new Tuple2<>(v._1, v._2._2));

        JavaPairRDD<String, Double> addd = vertex.subtract(toSet).mapToPair(v->new Tuple2<>(v, "")).join(graphTmp).mapToPair(v->new Tuple2<>(v._1, v._2._2)).values().
                flatMapToPair(v-> {
                    List<String> list = Lists.newArrayList(v);
                    List<Tuple2<String, Double>> res = new ArrayList<>();
                    for (String str: list)
                        res.add(new Tuple2<>(str, 1.0 / list.size()));
                    return res.iterator();
                }).reduceByKey((a,b)->a+b);

        JavaPairRDD<String, Iterable<String>> graph2 = graph.leftOuterJoin(addd).mapValues(v-> {
            List<String> list = Lists.newArrayList(v._1);
            list.add(String.valueOf(v._2.orElse(0.0)));
            list.add("1.0");
            return list;
        });

        JavaPairRDD<String, Double> graph3 = graph2.mapValues(v->{
            List<String> list = Lists.newArrayList(v);
            return Double.parseDouble(list.get(list.size()-1));
        });

        Double tmp = graph2.values().filter(p->Lists.newArrayList(p).get(0).equals("none")).map(v->Double.parseDouble(Lists.newArrayList(v).get(2))).reduce((a,b)->a+b) / vertex.count();
        Broadcast<Double> br = jsc.broadcast(tmp);

        graph2 = graph2.flatMapToPair(p->{
            List<String> list = Lists.newArrayList(p._2);
            List<Tuple2<String, String>> res = new ArrayList<>();
            if (list.get(0).equals("none")) {
                res.add(new Tuple2<>(p._1, "none"));
                res.add(new Tuple2<>(p._1, "add " + list.get(list.size()-2)));
                return res.iterator();
            }
            Double r = Double.parseDouble(list.get(list.size()-1));
            int size = list.size()-2;
            res.add(new Tuple2<>(p._1, "add " + list.get(list.size()-2)));
            for (int i=0; i<list.size()-2; i++) {
                res.add(new Tuple2<>(list.get(i), "rank " + String.valueOf(r/size)));
                res.add(new Tuple2<>(p._1, list.get(i)));
            }
            return res.iterator();
        }).groupByKey().mapValues(p-> {
            Double res = 0.0, addi = 0.0;
            List<String> list = new ArrayList<>();
            for (String str: p) {
                if (str.startsWith("rank")) res += Double.parseDouble(str.split(" ")[1]);
                else if (str.startsWith("add")) addi = Double.parseDouble(str.split(" ")[1]);
                else list.add(str);
            }
            res = res*0.85+0.15+0.85*br.value()+0.85*addi;
            list.add(addi.toString());
            list.add(res.toString());
            return list;
        });

        rankDif = graph2.mapValues(v->{
            List<String> list = Lists.newArrayList(v);
            return Double.parseDouble(list.get(list.size()-1));
        }).join(graph3).values().map(t->Math.abs(t._1 - t._2)).reduce((a,b)->a+b);

        graph3 = graph2.mapValues(v->{
            List<String> list = Lists.newArrayList(v);
            return Double.parseDouble(list.get(list.size()-1));
        });

        while(rankDif > epison) {
            // 上述实现为官方 github 里面的实现方法，但是在这个问题中
            // 如果使用上述实现方法，那么 join 操作就会产生多一个的 shuffle 操作，因而会导致每个循环内的运行时间增长
            // 所以将使用和 hadoop 相同的组织方式，即 from -> tolist + rank
            // 这样就可以减少一个 shuffle 的操作
            tmp = graph2.values().filter(p->Lists.newArrayList(p).get(0).equals("none")).map(v->Double.parseDouble(Lists.newArrayList(v).get(2))).reduce((a,b)->a+b) / vertex.count();
            Broadcast<Double> brTmp = jsc.broadcast(tmp);

            graph2 = graph2.flatMapToPair(p->{
                List<String> list = Lists.newArrayList(p._2);
                List<Tuple2<String, String>> res = new ArrayList<>();
                if (list.get(0).equals("none")) {
                    res.add(new Tuple2<>(p._1, "none"));
                    res.add(new Tuple2<>(p._1, "add " + list.get(list.size()-2)));
                    return res.iterator();
                }
                Double r = Double.parseDouble(list.get(list.size()-1));
                int size = list.size()-2;
                res.add(new Tuple2<>(p._1, "add " + list.get(list.size()-2)));
                for (int i=0; i<list.size()-2; i++) {
                    res.add(new Tuple2<>(list.get(i), "rank " + String.valueOf(r/size)));
                    res.add(new Tuple2<>(p._1, list.get(i)));
                }
                return res.iterator();
            }).groupByKey().mapValues(p-> {
                Double res = 0.0, addi = 0.0;
                List<String> list = new ArrayList<>();
                for (String str: p) {
                    if (str.startsWith("rank")) res += Double.parseDouble(str.split(" ")[1]);
                    else if (str.startsWith("add")) addi = Double.parseDouble(str.split(" ")[1]);
                    else list.add(str);
                }
                res = res*0.85+(0.15+0.85*brTmp.value())*(1+0.85*addi);
                list.add(addi.toString());
                list.add(res.toString());
                return list;
            });

            rankDif = graph2.mapValues(v->{
                List<String> list = Lists.newArrayList(v);
                return Double.parseDouble(list.get(list.size()-1));
            }).join(graph3).values().map(t->Math.abs(t._1 - t._2)).reduce((a,b)->a+b);

            graph3 = graph2.mapValues(v->{
                List<String> list = Lists.newArrayList(v);
                return Double.parseDouble(list.get(list.size()-1));
            });
        }
        Broadcast<Double> brTmp2 = jsc.broadcast(tmp);
        graph3 = vertex.subtract(toSet).mapToPair(v->new Tuple2<>(v, 0.15+0.85*brTmp2.value())).union(graph3).mapToPair(a -> new Tuple2<>(a._2, a._1)).sortByKey().mapToPair(a -> new Tuple2<>(a._2, a._1));
        graph3.coalesce(1).saveAsTextFile(output);

        // 62151
        System.out.println(System.currentTimeMillis()-start);
        jsc.close();
    }
}

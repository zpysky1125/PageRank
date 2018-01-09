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


public class sparkPageRankAverage {
    public static void main(String[] args) throws Exception{
        if(args.length < 3) {
            System.out.println("Error Input");
            System.exit(1);
        }
        long start = System.currentTimeMillis();
        String input = args[0], output = args[1];
        Double epison = Double.parseDouble(args[2]), rankDif;
        SparkConf conf = new SparkConf().setAppName("sparkPageRankAverage");
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

        JavaPairRDD<String, Iterable<String>> graph = vertex.subtract(fromSet).mapToPair(v->new Tuple2<>(v,"none")).union(pairs).groupByKey().cache();

        JavaPairRDD<String, Iterable<String>> graph2 = graph.mapValues(v-> {
            List<String> list = Lists.newArrayList(v);
            list.add("1.0");
            return list;
        });

        JavaPairRDD<String, Double> graph3 = graph2.mapValues(v->{
            List<String> list = Lists.newArrayList(v);
            return Double.parseDouble(list.get(list.size()-1));
        });

        rankDif = epison + 1;
        int iteration = 0;

        while(rankDif > epison) {
            iteration ++;

            Double tmp = graph2.values().filter(p->Lists.newArrayList(p).get(0).equals("none")).map(v->Double.parseDouble(Lists.newArrayList(v).get(1))).reduce((a,b)->a+b) / graph2.count();
            Broadcast<Double> br = jsc.broadcast(tmp);

            graph2 = graph2.flatMapToPair(p->{
                List<String> list = Lists.newArrayList(p._2);
                List<Tuple2<String, String>> res = new ArrayList<>();
                if (list.get(0).equals("none")) {
                    res.add(new Tuple2<>(p._1, "none"));
                    return res.iterator();
                }
                Double r = Double.parseDouble(list.get(list.size()-1));
                int size = list.size()-1;
                for (int i=0; i<list.size()-1; i++) {
                    res.add(new Tuple2<>(list.get(i), "rank " + String.valueOf(r/size)));
                    res.add(new Tuple2<>(p._1, list.get(i)));
                }
                return res.iterator();
            }).groupByKey().mapValues(p -> {
                Double res = 0.0;
                List<String> list = new ArrayList<>();
                for (String str: p) {
                    if (str.startsWith("rank")) res += Double.parseDouble(str.split(" ")[1]);
                    else list.add(str);
                }
                res = res*0.85+0.15+0.85*br.value();
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

        graph3 = graph3.mapToPair(a -> new Tuple2<>(a._2, a._1)).sortByKey().mapToPair(a -> new Tuple2<>(a._2, a._1));
        graph3.coalesce(1).saveAsTextFile(output);

        System.out.println(iteration);
        System.out.println(System.currentTimeMillis() - start);
        // 70
        // 230303
        jsc.close();
    }

}

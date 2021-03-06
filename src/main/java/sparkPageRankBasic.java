import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import org.apache.commons.lang.StringUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;


public class sparkPageRankBasic {
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

        JavaPairRDD<String, Double> addition = vertex.mapToPair(v -> new Tuple2<>(v, 0.15)).cache();
        JavaPairRDD<String, Double> rank = vertex.mapToPair(v -> new Tuple2<>(v, 1.0));
        JavaPairRDD<String, Double> lastRank = rank;

        // 如果一个点没有出边，增加一个到自己的出边，以此保证在矩阵乘法过程中，不会出现 rank 的弥散
        JavaPairRDD<String, Iterable<String>> graph = vertex.subtract(fromSet).mapToPair(v->new Tuple2<>(v,v)).union(pairs).groupByKey().cache();

        rankDif = epison + 1;

        int iteraion = 0;

        while(rankDif > epison) {
            iteraion ++;
            if (iteraion == 65) break;
            rank = rank.join(graph).values().flatMapToPair(line -> {
                List<Tuple2<String, Double>> res = new ArrayList<>();
                int size = Iterables.size(line._2);
                for (String str: line._2)
                    res.add(new Tuple2<>(str, 0.85 * line._1 / size));
                return res.iterator();
            }).union(addition).reduceByKey((a,b)->a+b);

            rankDif = rank.join(lastRank).values().map(t->Math.abs(t._1 - t._2)).reduce((a,b)->a+b);
            lastRank = rank;
        }
        rank = rank.mapToPair(a -> new Tuple2<>(a._2, a._1)).sortByKey().mapToPair(a -> new Tuple2<>(a._2, a._1));
        rank.coalesce(1).saveAsTextFile(output);
        // baisc 698406
        // no rankdif 419239
        System.out.println(System.currentTimeMillis()-start);
        jsc.close();
    }
}

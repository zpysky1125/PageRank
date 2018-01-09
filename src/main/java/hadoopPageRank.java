import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.*;
import java.util.*;

public class hadoopPageRank {

    public static class initGraphMapper extends Mapper<LongWritable, Text, Text, Text> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] lineSplit = value.toString().split("\t");
            lineSplit[0] = StringUtils.strip(lineSplit[0], "\""); lineSplit[1] = StringUtils.strip(lineSplit[1], "\"");
            if (lineSplit[1].equals("0") || lineSplit[0].equals("0")) return;
            context.write(new Text(lineSplit[0]), new Text(lineSplit[1]));
            context.write(new Text(lineSplit[0]), new Text("node"));
            context.write(new Text(lineSplit[1]), new Text("node"));
        }
    }

    public static class initGraphReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Set<String> set = new HashSet<>();
            for (Text text: values) {
                String val = text.toString();
                if (val.equals("node")) continue;
                set.add(val);
            }
            // if a node does not have out edges, then add a edge to itself
            if (set.size() > 0) context.write(key, new Text("1.0" + "\t" + String.join("\t", set)));
            else context.write(key, new Text("1.0" + "\t" + key.toString()));
        }
    }

    public static class pageRankMapper extends Mapper<LongWritable, Text, Text, Text> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] lineSplit = value.toString().split("\t");
            Double rank = Double.parseDouble(lineSplit[1]);
            int size = lineSplit.length - 2;
            for (int i=0; i<lineSplit.length; i++) {
                if (i == 0 || i == 1) continue;
                context.write(new Text(lineSplit[i]), new Text("rank" + " " + rank / size));
                context.write(new Text(lineSplit[0]), new Text(lineSplit[i]));
            }
        }
    }

    public static class pageRankReducer extends Reducer<Text, Text, Text, Text> {
        @Override
        protected void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            Double res = 0.0;
            List<String> list = new ArrayList<>();
            for (Text text: values) {
                String val = text.toString();
                if (val.startsWith("rank")) res += Double.parseDouble(val.split(" ")[1]);
                else list.add(val);
            }
            res = res * 0.85 + 0.15;
            context.write(key, new Text(res + "\t" + String.join("\t", list)));
        }
    }

    public static class resultMapper extends Mapper<LongWritable, Text, DoubleWritable, Text> {
        @Override
        protected void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            String[] lineSplit = value.toString().split("\t");
            context.write(new DoubleWritable(Double.parseDouble(lineSplit[1])), new Text(lineSplit[0]));
        }
    }

    public static class resultReducer extends Reducer<DoubleWritable, Text, Text, Text> {
        @Override
        protected void reduce(DoubleWritable key, Iterable<Text> values, Context context) throws IOException, InterruptedException {
            for (Text text: values)
                context.write(text, new Text(key.toString()));
        }
    }

    private static void initGraph(String input, String output) throws Exception{
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Hadoop Page Rank");
        job.setJarByClass(hadoopPageRank.class);
        job.setMapperClass(hadoopPageRank.initGraphMapper.class);
        job.setReducerClass(hadoopPageRank.initGraphReducer.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.setInputPaths(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));
        job.waitForCompletion(true);
    }

    private static void pageRankIteration(String input, String output) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Hadoop Page Rank");
        job.setJarByClass(hadoopPageRank.class);
        job.setMapperClass(hadoopPageRank.pageRankMapper.class);
        job.setReducerClass(hadoopPageRank.pageRankReducer.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.setInputPaths(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));
        job.waitForCompletion(true);
    }

    private static double getRankDif(String input1, String input2) throws Exception {
        FileSystem fs = FileSystem.get(new Configuration());
        File file1 = new File(input1 + "/part-r-00000");
        File file2 = new File(input2 + "/part-r-00000");
        BufferedReader reader = null;
        Map<String, Double> rank = new HashMap<>();
        String line = null;
        try {
            reader = new BufferedReader(new FileReader(file1));
            while ((line = reader.readLine()) != null) {
                String[] lineSplit = line.split("\t");
                rank.put(lineSplit[0], Double.parseDouble(lineSplit[1]));
            }
            reader.close();
            reader = new BufferedReader(new FileReader(file2));
            while ((line = reader.readLine()) != null) {
                String[] lineSplit = line.split("\t");
                rank.put(lineSplit[0], Math.abs(rank.get(lineSplit[0]) - Double.parseDouble(lineSplit[1])));
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e1) {
                    e1.printStackTrace();
                }
            }
        }
        return rank.values().stream().reduce((a,b)->a+b).get();
    }

    private static Integer pageRank(String arg, Double epison) throws Exception {
        Double rankDif = 1.0;
        int round = 0;
        String input = arg+round, output = arg + (++round);
        while (rankDif > epison) {
            pageRankIteration(input, output);
            rankDif = getRankDif(input, output);
            input = output; output = arg + (++round);
        }
        return --round;
    }

    private static void getResult(String input, String output) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Hadoop Page Rank");
        job.setJarByClass(hadoopPageRank.class);
        job.setMapperClass(hadoopPageRank.resultMapper.class);
        job.setReducerClass(hadoopPageRank.resultReducer.class);
        job.setMapOutputKeyClass(DoubleWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.setInputPaths(job, new Path(input));
        FileOutputFormat.setOutputPath(job, new Path(output));
        job.waitForCompletion(true);
    }

    // 为什么不使用 HashMap，因为 Hadoop 不支持全局变量，所以只能把 HashMap 保存成一个文件，然后通过 DistributedCache，然后将它传递给其它的节点。这样才能够将 HashMap 让所有节点都可知
    // 但是如果使用这种方法，不会比直接将聚合好的 from -> tolist，这种方式更加优秀，所以没有使用这种方法。而且在运行过程当中同样存在很多的文件读取问题，会对性能带来影响。
    // 这种使用 HashMap 的方法虽然在单机中是可行的，甚至效果优于常规方法，但是由于 HashMap 本身不能进行传递和共享，所以其实这种做法是错误的。

    public static void main(String[] args) throws Exception{
        if(args.length < 3) {
            System.out.println("Error Input");
            System.exit(1);
        }
        long start = System.currentTimeMillis();

        initGraph(args[0], args[1]+0);
        int round = pageRank(args[1], Double.parseDouble(args[2]));
        getResult(args[1] + round, args[1]+"result");
        // 164904
        System.out.println(System.currentTimeMillis()-start);
    }
}
